"""Checkpoint-only prediction for a new input-only graph HDF5 case."""
import argparse
from pathlib import Path
import h5py,numpy as np,torch
from .hdf5_data import read_graph_case
from .eq_descriptors_v1 import extract_eq_descriptors,GLOBAL_FEATURE_NAMES,WINDOW_FEATURE_NAMES
from .eq_recurrent_models_v2 import build_model

def predict(input_hdf5,case_id,checkpoint,output,torch_threads=1):
    torch.set_num_threads(torch_threads);g=read_graph_case(input_hdf5,case_id)
    eq_global,sequence=extract_eq_descriptors(g.ground_motion,g.time)
    cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
    gm,gs,nm,ns,em,es,qm,qs,ys=[np.asarray(x,dtype=np.float32) for x in cp["normalizer"]]
    recorded=g.metadata.get("recorded_node_ids",[])
    if not recorded:raise ValueError(f"{case_id}: missing first recorder node ID")
    found=np.flatnonzero(np.asarray(g.node_ids,dtype=np.int64)==int(recorded[0]))
    if len(found)!=1:raise ValueError(f"{case_id}: first recorder node is not unique in input graph")
    context={"global_features":torch.from_numpy((np.concatenate([g.global_features,eq_global]).astype(np.float32)-gm)/gs),
             "node_features":torch.from_numpy((g.node_features.astype(np.float32)-nm)/ns),
             "edge_index":torch.from_numpy(g.edge_index.astype(np.int64)).long(),
             "edge_features":torch.from_numpy((g.edge_features.astype(np.float32)-em)/es),
             "ground_motion":torch.from_numpy((sequence.astype(np.float32)-qm)/qs),"target_index":int(found[0])}
    model=build_model(cp["model"],cp["global_dim"],cp["node_dim"],cp["edge_dim"],cp["seq_dim"],hidden=cp["hidden"],**cp.get("model_hyperparameters",{}));model.load_state_dict(cp["state"]);model.eval()
    with torch.no_grad():pred=(model(context).numpy()*ys)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(out,"w") as h:
        h.attrs["model"]=cp["model"];h.attrs["target_node_id"]=int(recorded[0]);h.attrs["response_components"]="Ux,Uy";h.attrs["units"]="m,m"
        h.attrs["global_eq_feature_names_json"]=__import__("json").dumps(GLOBAL_FEATURE_NAMES);h.attrs["window_feature_names_json"]=__import__("json").dumps(WINDOW_FEATURE_NAMES)
        h.create_dataset("predicted_peak_abs_response",data=pred.astype(np.float32))
        h.create_dataset("eq_global_descriptors",data=eq_global);h.create_dataset("eq_window_descriptors",data=sequence)
    return pred

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-hdf5",required=True);p.add_argument("--case-id",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--torch-threads",type=int,default=1);a=p.parse_args();print(predict(a.input_hdf5,a.case_id,a.checkpoint,a.output,a.torch_threads))
if __name__=="__main__":main()
