"""Tune, five-fold train, and test six recurrent models plus GPR for first-node Ux/Uy peaks."""
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.chart import ScatterChart,Reference
from openpyxl.chart.series_factory import SeriesFactory
from openpyxl.chart.series import SeriesLabel
from .eq_recurrent_models_v2 import MODEL_NAMES as RECURRENT_MODEL_NAMES,build_model,model_hyperparameters
from .feature_shapley_v1 import feature_groups, feature_value_summaries, sampled_shapley, write_outputs as write_shapley_outputs
from .gpr_peak_v1 import vectorize_case, shapley_group_indices, predict_bundle, sampled_shapley_gpr
MODEL_NAMES=RECURRENT_MODEL_NAMES+('gpr',)

DOFS=("Ux","Uy")
COLORS=("#2468a2","#d87522")
MARKERS=("o","s","^","D","v","P","X")

def _read(db,cid):
    with h5py.File(db,"r") as h:
        c=h[cid];ids=c["node_ids"][:];rec=json.loads(c.attrs.get("recorded_node_ids","[]"))
        if not rec:raise ValueError(f"{cid}: no recorded_node_ids metadata")
        ix=np.flatnonzero(ids.astype(np.int64)==int(rec[0]))
        if len(ix)!=1:raise ValueError(f"{cid}: first recorder node missing or duplicated")
        target=c["target"][:].astype(np.float32);mask=c["mask"][:].astype(np.float32)
        if target.shape!=(2,) or mask.shape!=(2,):raise ValueError(f"{cid}: expected exactly [Ux, Uy] peak labels")
        return (np.concatenate([c["static"][:],c["eq_global"][:]]).astype(np.float32),
                c["node_features"][:].astype(np.float32),c["edge_index"][:].astype(np.int64),
                c["edge_features"][:].astype(np.float32),c["eq_sequence"][:].astype(np.float32),
                target,mask,float(ix[0]))

def _load_many(db,ids):return {cid:_read(db,cid) for cid in ids}

def _scale(x,axis=0):
    if x.size==0:
        width=x.shape[-1]
        return np.zeros(width,dtype=np.float32),np.ones(width,dtype=np.float32)
    mean=x.mean(axis=axis);std=x.std(axis=axis);std=np.where(std<1e-7,1.,std)
    return mean.astype(np.float32),std.astype(np.float32)

def _fit_norm(data,ids):
    gm,gs=_scale(np.stack([data[c][0] for c in ids]))
    nm,ns=_scale(np.concatenate([data[c][1] for c in ids],axis=0))
    em,es=_scale(np.concatenate([data[c][3] for c in ids],axis=0))
    qm,qs=_scale(np.concatenate([data[c][4] for c in ids],axis=0))
    ys=np.maximum(np.sqrt(np.mean(np.stack([data[c][5] for c in ids])**2,axis=0)),1e-8).astype(np.float32)
    return (gm,gs,nm,ns,em,es,qm,qs,ys)

def _tensor_context(case,norm):
    gm,gs,nm,ns,em,es,qm,qs,_=norm;g,n,ei,e,q,_,_,target_idx=case
    return {"global_features":torch.from_numpy((g-gm)/gs),"node_features":torch.from_numpy((n-nm)/ns),
            "edge_index":torch.from_numpy(ei).long(),"edge_features":torch.from_numpy((e-em)/es),
            "ground_motion":torch.from_numpy((q-qm)/qs),"target_index":int(target_idx)}

def _fit(payload):
    db,name,tr,va,args,ck,logpath=payload;torch.set_num_threads(args.torch_threads)
    data=_load_many(db,tr+va);norm=_fit_norm(data,tr);gm,gs,nm,ns,em,es,qm,qs,ys=norm
    tune_rng=np.random.default_rng(args.seed+sum(ord(c) for c in name));arch_keys=model_hyperparameters(name)
    base={"hidden":args.hidden,"recurrent_layers":2,"dropout":.05,"learning_rate":args.learning_rate,"weight_decay":1e-5}
    if "attention_heads" in arch_keys:base["attention_heads"]=4
    if "conv_kernel" in arch_keys:base["conv_kernel"]=3
    candidates=[base]
    while len(candidates)<args.tuning_trials:
        cfg={"hidden":int(tune_rng.choice([64,96,128])),"recurrent_layers":int(tune_rng.choice([1,2,3])),"dropout":float(tune_rng.choice([0.0,.05,.15])),"learning_rate":float(tune_rng.choice([3e-4,1e-3,3e-3])),"weight_decay":float(tune_rng.choice([1e-6,1e-5,1e-4]))}
        for key,choices in (("attention_heads",[2,4]),("conv_kernel",[3,5])):
            if key in arch_keys:cfg[key]=int(tune_rng.choice(choices))
        if cfg not in candidates:candidates.append(cfg)
    tuning_rows=[];chosen=None;chosen_score=float("inf")
    def train_config(config,max_epochs,stop_patience,seed,tag):
        torch.manual_seed(seed)
        model=build_model(name,len(gm),data[tr[0]][1].shape[1],data[tr[0]][3].shape[1],data[tr[0]][4].shape[1],hidden=config["hidden"],**{k:config[k] for k in arch_keys})
        opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=args.lr_factor,patience=args.lr_patience,min_lr=args.min_lr)
        best=float("inf");state=None;stale=0;epoch_rows=[];rng=np.random.default_rng(seed)
        for ep in range(1,max_epochs+1):
            model.train();order=list(tr);rng.shuffle(order);losses=[]
            for cid in order:
                case=data[cid];y=torch.from_numpy(case[5]/ys);mask=torch.from_numpy(case[6]);pred=model(_tensor_context(case,norm))
                loss=(torch.nn.functional.smooth_l1_loss(pred,y,reduction="none")*mask).sum()/mask.sum().clamp_min(1.)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            model.eval();vals=[]
            with torch.no_grad():
                for cid in va:
                    case=data[cid];pred=model(_tensor_context(case,norm));yy=torch.from_numpy(case[5]/ys);mm=torch.from_numpy(case[6])
                    vals.append(float((torch.nn.functional.smooth_l1_loss(pred,yy,reduction="none")*mm).sum()/mm.sum().clamp_min(1.)))
            vl=float(np.mean(vals)) if vals else float(np.mean(losses));scheduler.step(vl);lr=float(opt.param_groups[0]["lr"])
            epoch_rows.append({"phase":tag,"epoch":ep,"train_loss":float(np.mean(losses)),"validation_loss":vl,"learning_rate":lr})
            if tag=="final":print(f"{name} {tag} epoch {ep:03d}: train={np.mean(losses):.6g} val={vl:.6g} lr={lr:.3g}",flush=True)
            if vl<best-args.min_delta:best=vl;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=stop_patience:
                if tag=="final":print(f"{name}: early stopping at epoch {ep} (no validation improvement for {stop_patience} epochs)",flush=True)
                break
        return best,state,epoch_rows
    for trial,config in enumerate(candidates,1):
        score,_,_=train_config(config,args.tuning_epochs,args.tuning_patience,args.seed+trial,f"tune_{trial}")
        tuning_rows.append({"trial":trial,"validation_loss":score,**config});print(f"{name} tuning trial {trial}/{len(candidates)}: val={score:.6g}; config={config}",flush=True)
        if score<chosen_score:chosen_score=score;chosen=config
    print(f"{name}: selected hyperparameters by validation loss: {chosen}",flush=True)
    final_score,best_state,final_rows=train_config(chosen,args.epochs,args.patience,args.seed+10000,"final")
    Path(ck).parent.mkdir(parents=True,exist_ok=True);Path(logpath).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state":best_state,"global_dim":len(gm),"node_dim":data[tr[0]][1].shape[1],"edge_dim":data[tr[0]][3].shape[1],"seq_dim":data[tr[0]][4].shape[1],"hidden":chosen["hidden"],"model_hyperparameters":{k:chosen[k] for k in arch_keys},"hyperparameters":chosen,"normalizer":[x.tolist() for x in norm],"model":name},ck)
    all_logs=[{"phase":f"tune_trial_{r['trial']}","epoch":0,"train_loss":"","validation_loss":r["validation_loss"],"learning_rate":r["learning_rate"],"config_json":json.dumps({k:v for k,v in r.items() if k not in ("trial","validation_loss")},sort_keys=True)} for r in tuning_rows]
    all_logs += [{**r,"config_json":json.dumps(chosen,sort_keys=True)} for r in final_rows]
    with open(logpath,"w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=["phase","epoch","train_loss","validation_loss","learning_rate","config_json"]);w.writeheader();w.writerows(all_logs)
    Path(ck).with_suffix(".hyperparameters.json").write_text(json.dumps({"model":name,"selected_hyperparameters":chosen,"tuning_validation_loss":chosen_score,"refit_best_validation_loss":final_score,"trials":tuning_rows},indent=2),encoding="utf8")


def _fit_gpr(payload):
    """Tune and fit two Gaussian-process regressors using the same fold data."""
    import joblib
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    db,name,tr,va,args,ck,logpath=payload
    data=_load_many(db,tr+va)
    xraw={cid:vectorize_case(data[cid]) for cid in tr+va}
    scaler=StandardScaler().fit(np.stack([xraw[cid] for cid in tr]))
    x={cid:scaler.transform(xraw[cid].reshape(1,-1))[0] for cid in tr+va}
    target_scale=np.zeros(2,dtype=np.float64)
    for j in range(2):
        vals=np.asarray([data[cid][5][j] for cid in tr if data[cid][6][j]>0],dtype=np.float64)
        if not len(vals): raise ValueError(f"GPR has no training labels for {DOFS[j]}")
        target_scale[j]=max(float(np.sqrt(np.mean(vals**2))),1e-8)
    tune_rng=np.random.default_rng(args.seed+sum(ord(c) for c in name))
    candidates=[{"kernel":"matern_2.5","length_scale":1.0,"alpha":1e-6}]
    choices=[(kernel,length,alpha)
             for kernel in ("rbf","matern_1.5","matern_2.5")
             for length in (.1,.3,1.,3.,10.)
             for alpha in (1e-8,1e-6,1e-4,1e-2)]
    tune_rng.shuffle(choices)
    for kernel,length_scale,alpha in choices:
        cfg={"kernel":kernel,"length_scale":length_scale,"alpha":alpha}
        if cfg not in candidates and len(candidates)<args.tuning_trials:candidates.append(cfg)
        if len(candidates)>=args.tuning_trials:break

    def make_kernel(config):
        length=float(config["length_scale"])
        if config["kernel"]=="rbf":
            base=RBF(length_scale=length,length_scale_bounds=(1e-2,1e2))
        else:
            nu=1.5 if config["kernel"]=="matern_1.5" else 2.5
            base=Matern(length_scale=length,length_scale_bounds=(1e-2,1e2),nu=nu)
        return ConstantKernel(1.0,(1e-2,1e2))*base

    def fit_estimators(config,seed):
        estimators=[]
        for j in range(2):
            valid=[cid for cid in tr if data[cid][6][j]>0]
            estimator=GaussianProcessRegressor(kernel=make_kernel(config),alpha=float(config["alpha"]),
                normalize_y=True,n_restarts_optimizer=0,random_state=seed+j)
            estimator.fit(np.stack([x[cid] for cid in valid]),
                          np.asarray([data[cid][5][j]/target_scale[j] for cid in valid],dtype=np.float64))
            estimators.append(estimator)
        return estimators

    def score(estimators,ids):
        losses=[]
        for cid in ids:
            yy=data[cid][5]/target_scale;mask=data[cid][6].astype(bool)
            pred=np.zeros(2,dtype=np.float64)
            for j,estimator in enumerate(estimators):
                pred[j]=max(0.0,float(estimator.predict(x[cid].reshape(1,-1))[0]))
            error=np.abs(pred[mask]-yy[mask])
            losses.extend(np.where(error<1.,.5*error**2,error-.5).tolist())
        return float(np.mean(losses)) if losses else float("inf")

    tuning_rows=[];chosen=None;chosen_score=float("inf")
    # Limit BLAS threads so concurrent GPR jobs respect the Slurm CPU request.
    with threadpool_limits(limits=1):
        for trial,config in enumerate(candidates,1):
            estimators=fit_estimators(config,args.seed+trial)
            val_score=score(estimators,va)
            tuning_rows.append({"trial":trial,"validation_loss":val_score,**config})
            print(f"{name} tuning trial {trial}/{len(candidates)}: val={val_score:.6g}; config={config}",flush=True)
            if val_score<chosen_score:chosen_score=val_score;chosen=config
        estimators=fit_estimators(chosen,args.seed+10000)
        final_score=score(estimators,va)
        train_score=score(estimators,tr)
    bundle={"format":"SSI-ML-GPR-FirstNodePeak-First2-v1","model":"gpr",
            "estimators":estimators,"scaler":scaler,"target_scale":target_scale,
            "hyperparameters":chosen,"vector_schema":"global; node mean/std/min/max; edge mean/std/min/max; flattened ordered EQ windows"}
    Path(ck).parent.mkdir(parents=True,exist_ok=True);Path(logpath).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,ck,compress=3)
    with open(logpath,"w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=["phase","epoch","train_loss","validation_loss","learning_rate","config_json"]);w.writeheader()
        for row in tuning_rows:w.writerow({"phase":f"tune_trial_{row['trial']}","epoch":0,"train_loss":"","validation_loss":row["validation_loss"],"learning_rate":"","config_json":json.dumps({k:v for k,v in row.items() if k not in ("trial","validation_loss")},sort_keys=True)})
        w.writerow({"phase":"final_fit","epoch":0,"train_loss":train_score,"validation_loss":final_score,"learning_rate":"","config_json":json.dumps(chosen,sort_keys=True)})
    hp={"model":name,"selected_hyperparameters":chosen,"tuning_validation_loss":chosen_score,
        "refit_best_validation_loss":final_score,"trials":tuning_rows}
    Path(ck).with_suffix(".hyperparameters.json").write_text(json.dumps(hp,indent=2),encoding="utf8")
    print(f"{name}: selected hyperparameters by validation loss: {chosen}",flush=True)
    print(f"{name} final fit: train={train_score:.6g} val={final_score:.6g}; predictive uncertainty saved with checkpoint",flush=True)

def _valid_ape(actual, predicted, threshold):
    actual=np.asarray(actual,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.float64)
    valid=np.abs(actual)>=threshold
    ape=np.full(actual.shape,np.nan,dtype=np.float64)
    ape[valid]=np.abs(predicted[valid]-actual[valid])/np.abs(actual[valid])*100.0
    return ape,valid


def _make_plots(out,preds,metrics,mape_thresholds):
    """Save four response-specific scatter figures, each with only five cases."""
    out.mkdir(parents=True,exist_ok=True)
    # Prevent prior v15 combined plots from being mistaken for current figures.
    for old_name in ("eq_descriptor_Ux_actual_vs_predicted_scatter.png",
                     "eq_descriptor_Uy_actual_vs_predicted_scatter.png"):
        (out/old_name).unlink(missing_ok=True)
    selected_rows=[]
    for j,dof in enumerate(DOFS):
        threshold=mape_thresholds[dof]
        for selection in ("best5","worst5"):
            fig,ax=plt.subplots(1,1,figsize=(9,7))
            plotted=False
            for mi,name in enumerate(MODEL_NAMES):
                records=preds[name]
                eligible=[]
                for record in records:
                    cid,y,pred=record
                    ape,valid=_valid_ape([y[j]],[pred[j]],threshold)
                    if valid[0]:eligible.append((float(ape[0]),record))
                eligible.sort(key=lambda item:item[0])
                chosen=eligible[:5] if selection=="best5" else eligible[max(min(5,len(eligible)),len(eligible)-5):]
                for rank,(ape,(cid,y,pred)) in enumerate(chosen,1):
                    selected_rows.append({"model":name,"output":dof,"selection":selection,
                        "rank":rank,"simulation":cid,"actual_m":float(y[j]),
                        "predicted_m":float(pred[j]),"APE_percent":ape})
                if not chosen:continue
                actual=np.asarray([item[1][1][j] for item in chosen]);predicted=np.asarray([item[1][2][j] for item in chosen])
                ax.scatter(actual,predicted,s=48,alpha=.72,color=plt.get_cmap("tab10")(mi),marker=MARKERS[mi],label=f"{name} (n={len(chosen)})")
                plotted=True
            if plotted:
                low=min(float(ax.get_xlim()[0]),float(ax.get_ylim()[0]));high=max(float(ax.get_xlim()[1]),float(ax.get_ylim()[1]))
                ax.plot([low,high],[low,high],"k--",lw=1,label="1:1");ax.set_xlim(low,high);ax.set_ylim(low,high)
            title_group="best 5" if selection=="best5" else "worst 5"
            ax.set_title(f"Held-out {dof} peak displacement: {title_group} per model across folds")
            ax.set_xlabel(f"Actual max absolute {dof} displacement (m)");ax.set_ylabel(f"Predicted max absolute {dof} displacement (m)");ax.grid(alpha=.22)
            row_metrics={r["model"]:r for r in metrics}
            legend_order=sorted(range(len(MODEL_NAMES)),key=lambda i:row_metrics[MODEL_NAMES[i]][f"{dof}_MAPE_percent"] if np.isfinite(row_metrics[MODEL_NAMES[i]][f"{dof}_MAPE_percent"]) else float("inf"))
            labels=[]
            for i in legend_order:
                name=MODEL_NAMES[i]
                row=row_metrics[name]
                labels.append(plt.Line2D([],[],marker=MARKERS[i],linestyle="",color=plt.get_cmap("tab10")(i),
                    label=f"{name}: {dof} MAPE {row[f'{dof}_MAPE_percent']:.2f}% ± SD {row[f'{dof}_MAPE_std_percent']:.2f}%"))
            if plotted:labels.append(plt.Line2D([],[],color="black",linestyle="--",label="1:1"))
            ax.legend(handles=labels,fontsize=8,loc="best");fig.tight_layout()
            fig.savefig(out/f"eq_descriptor_{dof}_{selection}_actual_vs_predicted_scatter.png",dpi=160);plt.close(fig)
    with open(out/"best_worst_five_plot_data.csv","w",newline="",encoding="utf8") as f:
        fields=["model","output","selection","rank","simulation","actual_m","predicted_m","APE_percent"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected_rows)
    with open(out.parent/"overall_metrics.csv","w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    cols=("test_RMSE_macro_scaled","test_MAPE_macro","test_R2_macro");labels=("RMSE (scaled)","MAPE (%)","R²")
    vals=np.asarray([[r[c] for c in cols] for r in metrics],dtype=float);vals=np.nan_to_num(vals,nan=0.0,posinf=1.0,neginf=0.0)
    lo=np.nanmin(vals,axis=0);hi=np.nanmax(vals,axis=0)
    scaled=np.column_stack(((hi[0]-vals[:,0])/(hi[0]-lo[0]+1e-12),(hi[1]-vals[:,1])/(hi[1]-lo[1]+1e-12),(vals[:,2]-lo[2])/(hi[2]-lo[2]+1e-12)))
    angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist();angles+=angles[:1]
    fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,polar=True)
    for i,row in enumerate(metrics):
        v=scaled[i].tolist();ax.plot(angles,v+v[:1],label=f"{row['model']} (MAPE {row['overall_MAPE_percent']:.2f}% ± {row['overall_MAPE_std_percent']:.2f}%)");ax.fill(angles,v+v[:1],alpha=.025)
    ax.set_xticks(angles[:-1],labels);ax.set_ylim(0,1);ax.set_title("Held-out model metrics (normalized; higher is better)")
    ax.legend(bbox_to_anchor=(1.4,1.1),fontsize=7);fig.tight_layout();fig.savefig(out.parent/"training_prediction_spider.png",dpi=160,bbox_inches="tight");plt.close(fig)


def _write_prediction_workbook(out,all_predictions,metrics,hp_choices,hp_trials,gpr_uncertainties,mape_thresholds):
    """Export metrics, tuning and the best/worst five held-out cases per model."""
    book=Workbook();summary=book.active;summary.title="Overall_Metrics"
    fields=list(metrics[0]);summary.append(fields)
    for row in metrics:summary.append([row.get(k) for k in fields])
    summary.freeze_panes="A2";summary.auto_filter.ref=summary.dimensions
    for title,rows in (("Hyperparameter_Choices",hp_choices),("Tuning_Trials",hp_trials)):
        ws=book.create_sheet(title);cols=sorted({k for r in rows for k in r});ws.append(cols)
        for row in rows:ws.append([row.get(k) for k in cols])
        ws.freeze_panes="A2"
        if ws.max_row>1:ws.auto_filter.ref=ws.dimensions
    uncertainty=book.create_sheet("GPR_Predictive_Uncertainty")
    uncertainty.append(["Fold","Split","Simulation","Ux_predictive_std_m","Uy_predictive_std_m"])
    for fold,split,cid,_,_ in all_predictions.get("gpr",[]):
        std=gpr_uncertainties.get((fold,split,cid),[float("nan"),float("nan")])
        uncertainty.append([fold,split,cid,float(std[0]),float(std[1])])
    uncertainty.freeze_panes="A2"
    if uncertainty.max_row>1:uncertainty.auto_filter.ref=uncertainty.dimensions
    selected=book.create_sheet("Best_Worst_Selection");selected.append(["Model","Output","Rank","Selection","Fold","Simulation","Actual_m","Predicted_m","APE_percent"])
    for name in MODEL_NAMES:
        tests=[]
        for fold,split,cid,y,pred in all_predictions[name]:
            if split=="test":
                component_apes=[]
                for j,dof in enumerate(DOFS):
                    ape,valid=_valid_ape([y[j]],[pred[j]],mape_thresholds[dof])
                    component_apes.append(float(ape[0]) if valid[0] else np.nan)
                valid_values=[value for value in component_apes if np.isfinite(value)]
                if valid_values:tests.append((float(np.mean(valid_values)),fold,cid,y,pred,component_apes))
        tests.sort(key=lambda r:r[0]);n=len(tests);k=min(5,n)
        chosen=[("Best",i+1,r) for i,r in enumerate(tests[:k])]+[("Worst",i+1,r) for i,r in enumerate(tests[max(k,n-5):])]
        for j,dof in enumerate(DOFS):
            ranked=[]
            for fold,split,cid,y,pred in all_predictions[name]:
                if split!="test":continue
                ape,valid=_valid_ape([y[j]],[pred[j]],mape_thresholds[dof])
                if valid[0]:ranked.append((float(ape[0]),fold,cid,y,pred))
            ranked.sort(key=lambda item:item[0]);count=min(5,len(ranked))
            chosen_dof=[("Best",i+1,row) for i,row in enumerate(ranked[:count])]+[("Worst",i+1,row) for i,row in enumerate(ranked[max(count,len(ranked)-5):])]
            for selection,rank,(ape,fold,cid,y,pred) in chosen_dof:
                # The selection sheet contains the response-specific selections
                # used in the four scatter plots.
                selected.append([name,dof,rank,selection,fold,cid,float(y[j]),float(pred[j]),ape])
        ws=book.create_sheet(name[:31]);metric=next(x for x in metrics if x["model"]==name)
        ws.append(["Model",name]);ws.append(["Overall held-out MAPE mean ± SD (%)",f"{metric['overall_MAPE_percent']:.6g} ± {metric['overall_MAPE_std_percent']:.6g}"])
        ws.append(["Best/worst cases use the mean available Ux/Uy APE; near-zero response components are left blank."])
        ws.append(["Selection","Rank","Fold","Simulation","UxUy_MAPE_percent","Ux_Actual","Ux_Predicted","Ux_Absolute_Error","Ux_APE_percent","Uy_Actual","Uy_Predicted","Uy_Absolute_Error","Uy_APE_percent"])
        for group,rank,(mape,fold,cid,y,pred,apes) in chosen:
            values=[group,rank,fold,cid,mape]
            for j in range(2):
                err=float(abs(pred[j]-y[j]));ape=float(apes[j]) if np.isfinite(apes[j]) else None
                values.extend([float(y[j]),float(pred[j]),err,ape])
            ws.append(values)
        ws.freeze_panes="F5";ws.auto_filter.ref=f"A4:M{max(4,ws.max_row)}"
        for j,dof in enumerate(DOFS):
            chart=ScatterChart();chart.title=f"{name}: {dof} actual vs predicted | overall MAPE {metric['overall_MAPE_percent']:.2f}% ± SD {metric['overall_MAPE_std_percent']:.2f}%"
            chart.x_axis.title=f"Actual peak {dof} displacement (m)";chart.y_axis.title=f"Predicted peak {dof} displacement (m)";chart.style=13;chart.height=7;chart.width=12;chart.legend.position="b"
            actual_col=6+4*j;pred_col=actual_col+1
            series=SeriesFactory(Reference(ws,min_col=pred_col,min_row=5,max_row=max(5,ws.max_row)),Reference(ws,min_col=actual_col,min_row=5,max_row=max(5,ws.max_row)),title_from_data=False)
            series.tx=SeriesLabel(v=dof);chart.series.append(series)
            ws.add_chart(chart,"O2" if j==0 else "O18")
    selected.freeze_panes="A2";selected.auto_filter.ref=selected.dimensions
    book.calculation.fullCalcOnLoad=True
    path=out/"best_worst_predictions_all_folds_all_models.xlsx";book.save(path)
    print(f"Excel workbook (5 best/5 worst across all folds and seven models): {path}",flush=True)


def run(a):
    if a.workers<1 or a.torch_threads<1: raise ValueError("workers and torch-threads must be positive")
    if a.shap_samples_per_fold<1 or a.shap_permutations<1: raise ValueError("SHAP sample/permutation counts must be positive")
    if min(a.epochs,a.patience,a.tuning_trials,a.tuning_epochs,a.tuning_patience)<1: raise ValueError("epochs, patience, tuning-trials, tuning-epochs, and tuning-patience must be positive")
    torch.set_num_threads(a.torch_threads)
    with h5py.File(a.database,"r") as h: ids=sorted(k for k in h.keys() if k.startswith("simulation_") and k.rsplit("_",1)[-1].isdigit())
    if len(ids)<a.folds: raise ValueError(f"Need at least {a.folds} cases; found {len(ids)}")
    shap_groups=feature_groups(a.database,ids[0]);shapley_samples={n:[] for n in MODEL_NAMES}
    rng=np.random.default_rng(a.seed); perm=list(ids);rng.shuffle(perm); folds=[perm[i::a.folds] for i in range(a.folds)]
    out=Path(a.output); jobs=[]; splits=[]
    for fi,test in enumerate(folds,1):
        pool=[c for c in perm if c not in test];rng.shuffle(pool);nv=max(1,int(round(len(pool)*a.validation_fraction))) if len(pool)>3 else 0; val=pool[:nv];tr=pool[nv:]
        if not val: val=tr[:1]; tr=tr[1:]
        if not tr: raise ValueError("No training cases after fold/validation split")
        print(f"Fold {fi}/{a.folds}: train={len(tr)}, validation={len(val)}, test={len(test)}",flush=True);splits.append((fi,tr,val,test))
        for name in MODEL_NAMES:
            checkpoint_ext=".joblib" if name=="gpr" else ".pt"
            jobs.append((a.database,name,tr,val,a,str(out/"checkpoints"/f"fold_{fi}"/f"{name}{checkpoint_ext}"),str(out/"logs"/f"fold_{fi}_{name}.csv")))
    if a.workers==1:
        for job in jobs:(_fit_gpr if job[1]=="gpr" else _fit)(job)
    else:
        print(f"Training {len(jobs)} fold/model jobs using {a.workers} worker processes",flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(_fit_gpr if j[1]=="gpr" else _fit,j) for j in jobs]
            for f in as_completed(futures): f.result()
    hp_choices=[];hp_trials=[]
    for fi,_,_,_ in splits:
        for name in MODEL_NAMES:
            hp_path=out/"checkpoints"/f"fold_{fi}"/f"{name}.hyperparameters.json"
            info=json.loads(hp_path.read_text(encoding="utf8"));hp_choices.append({"fold":fi,"model":name,"tuning_best_validation_loss":info["tuning_validation_loss"],"refit_best_validation_loss":info["refit_best_validation_loss"],**info["selected_hyperparameters"]})
            for trial in info["trials"]:hp_trials.append({"fold":fi,"model":name,**trial})
    if hp_choices:
        for filename,rows in (("hyperparameter_selections.csv",hp_choices),("hyperparameter_search_trials.csv",hp_trials)):
            with open(out/filename,"w",newline="",encoding="utf8") as f:
                fields=sorted({k for r in rows for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    predictions={n:[] for n in MODEL_NAMES};all_predictions={n:[] for n in MODEL_NAMES};gpr_uncertainties={}
    first_case=_read(a.database,ids[0])
    gpr_group_indices=shapley_group_indices(shap_groups,first_case[0].size,first_case[1].shape[1],
        first_case[3].shape[1],first_case[4].shape[0],first_case[4].shape[1])
    for fi,tr,val,test in splits:
        shap_rng=np.random.default_rng(a.seed+fi*1009)
        shap_ids=set(np.asarray(test,dtype=object)[shap_rng.choice(len(test),size=min(a.shap_samples_per_fold,len(test)),replace=False)].tolist())
        for name in MODEL_NAMES:
            if name=="gpr":
                import joblib
                bundle=joblib.load(out/"checkpoints"/f"fold_{fi}"/"gpr.joblib")
                for split,cases in (("validation",val),("test",test)):
                    for cid in cases:
                        case=_read(a.database,cid);xraw=vectorize_case(case)
                        p,std=predict_bundle(bundle,xraw);y=case[5]
                        all_predictions[name].append((fi,split,cid,y,p))
                        gpr_uncertainties[(fi,split,cid)]=std
                        if split=="test":
                            predictions[name].append((cid,y,p))
                            if cid in shap_ids:
                                rng=np.random.default_rng(a.seed+fi*100000+sum(ord(ch) for ch in name)+int(cid.split("_")[-1]))
                                phi=sampled_shapley_gpr(bundle,xraw,gpr_group_indices,a.shap_permutations,rng)
                                shapley_samples[name].append((fi,cid,phi,feature_value_summaries(case,shap_groups)))
                continue
            cp=torch.load(out/"checkpoints"/f"fold_{fi}"/f"{name}.pt",map_location="cpu",weights_only=False);norm=[np.asarray(x,dtype=np.float32) for x in cp["normalizer"]];ys=norm[-1]
            model=build_model(name,cp["global_dim"],cp["node_dim"],cp["edge_dim"],cp["seq_dim"],hidden=cp["hidden"],**cp.get("model_hyperparameters",{}));model.load_state_dict(cp["state"]);model.eval()
            for split,cases in (("validation",val),("test",test)):
                for cid in cases:
                    case=_read(a.database,cid)
                    context=_tensor_context(case,norm)
                    with torch.no_grad(): p=(model(context).numpy()*ys)
                    y=case[5]
                    all_predictions[name].append((fi,split,cid,y,p))
                    if split=="test":
                        predictions[name].append((cid,y,p))
                        if cid in shap_ids:
                            rng=np.random.default_rng(a.seed+fi*100000+sum(ord(ch) for ch in name)+int(cid.split("_")[-1]))
                            phi=sampled_shapley(model,context,shap_groups,ys,a.shap_permutations,rng)
                            input_values=feature_value_summaries(case,shap_groups)
                            shapley_samples[name].append((fi,cid,phi,input_values))
    # MAPE is undefined at zero and unstable close to zero. Ignore true
    # near-zero response labels using one documented output-specific cutoff:
    # 5% of the median nonzero absolute held-out response.
    mape_thresholds={}
    for i,dof in enumerate(DOFS):
        actual=np.asarray([y[i] for records in predictions.values() for _,y,_ in records],dtype=np.float64)
        nonzero=np.abs(actual[np.abs(actual)>1e-12])
        mape_thresholds[dof]=max(1e-8,0.05*float(np.median(nonzero))) if nonzero.size else 1e-8
    metrics=[]
    for name,records in predictions.items():
        case_mape=[]
        for _,y,pred in records:
            case_values=[]
            for i,dof in enumerate(DOFS):
                ape,valid=_valid_ape([y[i]],[pred[i]],mape_thresholds[dof])
                if valid[0]:case_values.append(float(ape[0]))
            if case_values:case_mape.append(float(np.mean(case_values)))
        row={"model":name,"test_simulations":len(records),
             "overall_MAPE_percent":float(np.mean(case_mape)) if case_mape else float("nan"),
             "overall_MAPE_std_percent":float(np.std(case_mape,ddof=1)) if len(case_mape)>1 else 0.0,
             "MAPE_cases_included":len(case_mape),
             "MAPE_cases_excluded_all_near_zero":len(records)-len(case_mape),
             "Ux_MAPE_threshold_m":mape_thresholds["Ux"],
             "Uy_MAPE_threshold_m":mape_thresholds["Uy"]}
        component_r2=[];component_rmse=[]
        for i,dof in enumerate(DOFS):
            yy=np.asarray([y[i] for _,y,_ in records]);pp=np.asarray([p[i] for _,_,p in records]);err=pp-yy
            rmse=float(np.sqrt(np.mean(err**2)));variance=float(np.sum((yy-yy.mean())**2));r2=float(1-np.sum(err**2)/variance) if variance>1e-15 else float("nan")
            ape,valid=_valid_ape(yy,pp,mape_thresholds[dof]);valid_ape=ape[valid]
            row[f"{dof}_RMSE"] = rmse
            row[f"{dof}_MAPE_percent"]=float(np.mean(valid_ape)) if valid_ape.size else float("nan")
            row[f"{dof}_MAPE_std_percent"]=float(np.std(valid_ape,ddof=1)) if valid_ape.size>1 else 0.0
            row[f"{dof}_MAPE_cases_included"]=int(valid.sum())
            row[f"{dof}_MAPE_cases_excluded_near_zero"]=int((~valid).sum())
            row[f"{dof}_R2"]=r2
            component_rmse.append(rmse);component_r2.append(r2)
        scales=np.maximum(np.std(np.asarray([y for _,y,_ in records]),axis=0),1e-12)
        row["test_RMSE_macro_scaled"]=float(np.mean(np.asarray(component_rmse)/scales));row["test_MAPE_macro"]=float(np.nanmean([row[f"{d}_MAPE_percent"] for d in DOFS]));row["test_R2_macro"]=float(np.nanmean(component_r2))
        metrics.append(row)
    _make_plots(out/"plots",predictions,metrics,mape_thresholds)
    write_shapley_outputs(out/"shapley",shapley_samples,shap_groups,a.shap_permutations)
    uncertainty_path=out/"gpr_predictive_uncertainty.csv"
    with uncertainty_path.open("w",newline="",encoding="utf8") as f:
        fields=["fold","split","simulation","Ux_predictive_std_m","Uy_predictive_std_m"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for fold,split,cid,_,_ in all_predictions["gpr"]:
            std=gpr_uncertainties.get((fold,split,cid),[float("nan"),float("nan")])
            w.writerow({"fold":fold,"split":split,"simulation":cid,"Ux_predictive_std_m":float(std[0]),"Uy_predictive_std_m":float(std[1])})
    with open(out/"all_folds_all_models_predictions.csv","w",newline="",encoding="utf8") as f:
        fields=["fold","split","model","simulation"]+[f"{x}_{s}" for x in DOFS for s in ("actual","predicted")]+["Ux_predictive_std_m","Uy_predictive_std_m"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,records in all_predictions.items():
            for fold,split,cid,y,pred in records:
                row={"fold":fold,"split":split,"model":name,"simulation":cid,"Ux_predictive_std_m":"","Uy_predictive_std_m":""}
                if name=="gpr":
                    std=gpr_uncertainties.get((fold,split,cid),[float("nan"),float("nan")]);row["Ux_predictive_std_m"]=float(std[0]);row["Uy_predictive_std_m"]=float(std[1])
                for i,dof in enumerate(DOFS):row[f"{dof}_actual"]=float(y[i]);row[f"{dof}_predicted"]=float(pred[i])
                w.writerow(row)
    with open(out/"heldout_predictions.csv","w",newline="",encoding="utf8") as f:
        fields=["fold","model","simulation"]+[f"{x}_{s}" for x in DOFS for s in ("actual","predicted")]+["Ux_predictive_std_m","Uy_predictive_std_m"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,records in predictions.items():
            for fold,split,cid,y,pred in all_predictions[name]:
                if split!="test":continue
                row={"fold":fold,"model":name,"simulation":cid,"Ux_predictive_std_m":"","Uy_predictive_std_m":""}
                if name=="gpr":
                    std=gpr_uncertainties.get((fold,split,cid),[float("nan"),float("nan")]);row["Ux_predictive_std_m"]=float(std[0]);row["Uy_predictive_std_m"]=float(std[1])
                for i,dof in enumerate(DOFS):row[f"{dof}_actual"]=float(y[i]);row[f"{dof}_predicted"]=float(pred[i])
                w.writerow(row)
    _write_prediction_workbook(out,all_predictions,metrics,hp_choices,hp_trials,gpr_uncertainties,mape_thresholds)
    print(f"Done. Scatter plot, overall MAPE ± SD, and CV predictions saved in {out}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--database",required=True);p.add_argument("--output",default="Outputs_ML/eq_descriptor_comparison_first2_v2");p.add_argument("--folds",type=int,default=5);p.add_argument("--epochs",type=int,default=300);p.add_argument("--patience",type=int,default=18);p.add_argument("--lr-patience",type=int,default=5);p.add_argument("--lr-factor",type=float,default=.5);p.add_argument("--min-lr",type=float,default=1e-6);p.add_argument("--min-delta",type=float,default=1e-6);p.add_argument("--hidden",type=int,default=96);p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--tuning-trials",type=int,default=4);p.add_argument("--tuning-epochs",type=int,default=40);p.add_argument("--tuning-patience",type=int,default=8);p.add_argument("--validation-fraction",type=float,default=.15);p.add_argument("--seed",type=int,default=42);p.add_argument("--workers",type=int,default=1);p.add_argument("--torch-threads",type=int,default=1);p.add_argument("--shap-samples-per-fold",type=int,default=3,help="Held-out simulations sampled per fold for approximate Shapley explanations");p.add_argument("--shap-permutations",type=int,default=8,help="Random feature orderings per held-out case; larger values are slower but more stable");run(p.parse_args())
if __name__=="__main__":main()
