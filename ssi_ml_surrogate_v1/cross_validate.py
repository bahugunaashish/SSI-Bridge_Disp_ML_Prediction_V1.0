"""Five-fold CV for per-simulation first-recorder-node six-DOF peak labels."""
import argparse,csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from .hdf5_data import list_cases,read_peak_case
from .models import MODEL_NAMES,build_model
from .train import tensor_case,load_dims,first_node_index,fit_input_normalizer,fit_target_scale

NAMES=("Ux","Uy","Uz","Rx","Ry","Rz")
def _loss(pred,y): return torch.nn.functional.smooth_l1_loss(pred,y)
def _fit(payload):
    db,name,train_ids,val_ids,args,checkpoint,normalizer,target_scale=payload; torch.set_num_threads(args.torch_threads); torch.manual_seed(args.seed)
    d=load_dims(db); model=build_model(name,node_dim=d[0],edge_dim=d[1],global_dim=d[2],motion_dim=d[3],response_dim=6,hidden=args.hidden)
    target_scale_t=torch.as_tensor(target_scale,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5); rng=np.random.default_rng(args.seed); best=float("inf"); state=None
    for ep in range(1,args.epochs+1):
        model.train(); losses=[]; order=list(train_ids); rng.shuffle(order)
        for cid in order:
            g,y,m=read_peak_case(db,cid); p=model(tensor_case(g,normalizer))[first_node_index(g)]; y_t=torch.from_numpy(y)/target_scale_t; m_t=torch.from_numpy(m); loss=(torch.nn.functional.smooth_l1_loss(p,y_t,reduction="none")*m_t).sum()/m_t.sum().clamp_min(1.)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss.detach()))
        model.eval(); vals=[]
        with torch.no_grad():
            for cid in val_ids:
                g,y,m=read_peak_case(db,cid); p=model(tensor_case(g,normalizer))[first_node_index(g)]; y_t=torch.from_numpy(y)/target_scale_t; m_t=torch.from_numpy(m); vals.append(float((torch.nn.functional.smooth_l1_loss(p,y_t,reduction="none")*m_t).sum()/m_t.sum().clamp_min(1.)))
        vl=float(np.mean(vals)) if vals else float(np.mean(losses)); print(f"{name} epoch {ep:03d}: train_normalized={np.mean(losses):.6g} val_normalized={vl:.6g}",flush=True)
        if vl<best: best=vl; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); checkpoint=Path(checkpoint); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":state,"model_name":name,"dims":d,"hidden":args.hidden,"input_normalizer":normalizer,"target_scale":target_scale},checkpoint)

def _scores(actual,pred,scale):
    err=pred-actual; rmse=np.sqrt(np.mean(err**2,axis=0)); mae=np.mean(abs(err),axis=0)
    mpae=np.mean(abs(err)/np.maximum(abs(actual),1e-8),axis=0)*100
    var=np.sum((actual-actual.mean(axis=0))**2,axis=0); r2=np.where(var>1e-15,1-np.sum(err**2,axis=0)/np.maximum(var,1e-15),np.nan)
    return rmse,mae,mpae,r2

def _save_best_worst_plots(out, heldout):
    """Write one compact per-model figure with translation/rotation separated."""
    plot_dir=out/"plots"; plot_dir.mkdir(parents=True,exist_ok=True)
    for name, cases in heldout.items():
        # Rank on six-DOF case MAPE. Each translation/rotation panel reports
        # its own three-DOF MAPE so metres and radians stay clearly separated.
        ranked=[]
        for cid,actual,pred in cases:
            dof_mape=np.abs(pred-actual)/np.maximum(np.abs(actual),1e-8)*100
            ranked.append((float(np.mean(dof_mape)),cid,actual,pred))
        ranked.sort(key=lambda item:item[0])
        best=ranked[:min(5,len(ranked))]
        worst=ranked[-min(5,len(ranked)):]
        groups=(("Best 5 — translations",best,slice(0,3),"Peak displacement (m)"),
                ("Worst 5 — translations",worst,slice(0,3),"Peak displacement (m)"),
                ("Best 5 — rotations",best,slice(3,6),"Peak rotation (rad)"),
                ("Worst 5 — rotations",worst,slice(3,6),"Peak rotation (rad)"))
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for ax,(title,selected,dof_slice,ylabel) in zip(axes.flat,groups):
            comps=NAMES[dof_slice]
            x=np.arange(len(selected)); width=.34
            actual=np.asarray([item[2][dof_slice] for item in selected])
            pred=np.asarray([item[3][dof_slice] for item in selected])
            for j,comp in enumerate(comps):
                offset=(j-(len(comps)-1)/2)*width
                ax.bar(x+offset-width/2,actual[:,j],width,label=f"Actual {comp}" if ax is axes[0,0] else None,color=("#3b6fb6" if j==0 else "#5a9b75" if j==1 else "#ad6eaa"))
                ax.bar(x+offset+width/2,pred[:,j],width,label=f"Predicted {comp}" if ax is axes[0,0] else None,color=("#91b4e3" if j==0 else "#9ec8ae" if j==1 else "#d1a4cf"))
            case_mape=[np.mean(np.abs(item[3][dof_slice]-item[2][dof_slice])/
                               np.maximum(np.abs(item[2][dof_slice]),1e-8))*100
                       for item in selected]
            ax.set_xticks(x,[f"{item[1]}\nMAPE {mp:.1f}%" for item,mp in zip(selected,case_mape)],rotation=0)
            ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y",alpha=.25)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=6,fontsize=8)
        fig.suptitle(f"{name}: held-out peak predictions (cases ranked by six-DOF MAPE)",y=.99)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(plot_dir/f"{name}_best_worst_comparison.png",dpi=140)
        plt.close(fig)

def run(a):
    if a.workers<1 or a.torch_threads<1: raise ValueError("workers and torch-threads must be positive")
    ids=list_cases(a.database)
    if len(ids)<a.folds: raise ValueError(f"Need at least {a.folds} complete simulations; found {len(ids)}")
    rng=np.random.default_rng(a.seed); shuffled=list(ids); rng.shuffle(shuffled); folds=[shuffled[i::a.folds] for i in range(a.folds)]
    out=Path(a.output); ck=out/"checkpoints"; out.mkdir(parents=True,exist_ok=True); jobs=[]; splits=[]
    for fi,test in enumerate(folds,1):
        pool=[c for c in shuffled if c not in test]; rng.shuffle(pool); nv=max(0,int(round(len(pool)*a.validation_fraction))) if len(pool)>=3 else 0
        val=pool[:nv]; train=pool[nv:]
        if not train: raise ValueError(f"Fold {fi} has no training simulations")
        splits.append((fi,test,train)); print(f"Fold {fi}/{a.folds}: train={len(train)}, validation={len(val)}, test={len(test)}",flush=True)
        normalizer=fit_input_normalizer(a.database,train); target_scale=fit_target_scale(a.database,train)
        print(f"Fold {fi}: fitted input and six-output scalers from training cases only",flush=True)
        for name in MODEL_NAMES: jobs.append((a.database,name,train,val,a,str(ck/f"fold_{fi}"/f"{name}.pt"),normalizer,target_scale))
    if a.workers==1:
        for job in jobs:_fit(job)
    else:
        print(f"Training {len(jobs)} fold/model jobs with {a.workers} processes",flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(_fit,j) for j in jobs]
            for f in as_completed(futures):f.result()
    by_model={m:{"train_y":[],"train_p":[],"test_y":[],"test_p":[]} for m in MODEL_NAMES}; rows=[]
    heldout={m:[] for m in MODEL_NAMES}
    all_labels=np.asarray([read_peak_case(a.database,c)[1] for c in ids]); scales=np.maximum(all_labels.std(axis=0),1e-12)
    for fi,test,train in splits:
        models={}
        for name in MODEL_NAMES:
            cp=torch.load(ck/f"fold_{fi}"/f"{name}.pt",map_location="cpu"); d=cp["dims"]
            model=build_model(name,node_dim=d[0],edge_dim=d[1],global_dim=d[2],motion_dim=d[3],response_dim=6,hidden=a.hidden);model.load_state_dict(cp["model"]);model.eval();models[name]=model
        for split,cases in (("train",train),("test",test)):
            for cid in cases:
                case_predictions={}
                g,y,_=read_peak_case(a.database,cid)
                for name,model in models.items():
                    with torch.no_grad(): p=(model(tensor_case(g,cp.get("input_normalizer")))[first_node_index(g)]*torch.as_tensor(cp.get("target_scale",[1.]*6))).numpy()
                    by_model[name][split+"_y"].append(y);by_model[name][split+"_p"].append(p)
                    if split=="test": heldout[name].append((cid,y.copy(),p.copy()))
                    if split=="test": case_predictions[name]=p
                    row={"fold":fi,"simulation":cid,"split":split,"model":name}
                    for i,ch in enumerate(NAMES):row.update({f"{ch}_actual_peak":float(y[i]),f"{ch}_predicted_peak":float(p[i]),f"{ch}_absolute_error":float(abs(y[i]-p[i]))})
                    rows.append(row)
        print(f"Completed fold {fi} held-out cases",flush=True)
    _save_best_worst_plots(out,heldout)
    # Per-component metrics and macro summaries. MPAE uses a 1e-8 floor for near-zero labels.
    summaries=[]
    for name,v in by_model.items():
        s={"model":name}
        for split in ("train","test"):
            y=np.asarray(v[split+"_y"]);p=np.asarray(v[split+"_p"])
            rm,ma,mp,r2=_scores(y,p,None)
            for j,ch in enumerate(NAMES):s.update({f"{split}_{ch}_RMSE":float(rm[j]),f"{split}_{ch}_MPAE":float(mp[j]),f"{split}_{ch}_R2":float(r2[j])})
            s[f"{split}_RMSE_macro"] = float(np.mean(rm/scales))
            s[f"{split}_MPAE_macro"] = float(np.mean(mp))
            s[f"{split}_R2_macro"] = float(np.nanmean(r2))
        summaries.append(s)
    fields=list(rows[0]);
    with (out/"metrics_by_simulation.csv").open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (out/"metrics_summary.csv").open("w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    # Radar: each metric normalized across all model/split values (lower error is reversed).
    labels=("RMSE","MPAE","R²"); fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,polar=True); ang=np.linspace(0,2*np.pi,3,endpoint=False).tolist(); ang+=ang[:1]
    raw={}
    for row in summaries:
        for split in ("train","test"):
            raw[(row["model"],split)]=[row[f"{split}_RMSE_macro"],row[f"{split}_MPAE_macro"],row[f"{split}_R2_macro"]]
    mins=[min(v[i] for v in raw.values()) for i in range(3)]; maxs=[max(v[i] for v in raw.values()) for i in range(3)]
    norm={k:[(maxs[i]-v[i])/(maxs[i]-mins[i]+1e-12) if i<2 else (v[i]-mins[i])/(maxs[i]-mins[i]+1e-12) for i in range(3)] for k,v in raw.items()}
    for name in MODEL_NAMES:
        for split,ls in (("train","--"),("test","-")):
            val=norm[(name,split)]; ax.plot(ang,val+val[:1],ls,label=f"{name} {split}",alpha=.85); ax.fill(ang,val+val[:1],alpha=.025)
    ax.set_xticks(ang[:-1],labels);ax.set_ylim(0,1);ax.set_title("First-node peak metrics (normalized; higher is better)");ax.legend(bbox_to_anchor=(1.3,1.12),fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/"training_prediction_spider.png",dpi=170,bbox_inches="tight");plt.close(fig)
    print(f"Wrote peak metrics, fold plots, summary, and training_prediction_spider.png under {out}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--database",required=True);p.add_argument("--output",default="results/first_node_peak_model_comparison");p.add_argument("--folds",type=int,default=5);p.add_argument("--epochs",type=int,default=10);p.add_argument("--hidden",type=int,default=96);p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--validation-fraction",type=float,default=.15);p.add_argument("--seed",type=int,default=42);p.add_argument("--workers",type=int,default=1);p.add_argument("--torch-threads",type=int,default=1);run(p.parse_args())
if __name__=="__main__":main()
