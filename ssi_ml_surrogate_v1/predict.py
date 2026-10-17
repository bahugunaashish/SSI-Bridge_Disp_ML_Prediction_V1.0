"""Predict first-recorder-node max-absolute six-DOF response peaks."""
import argparse,json
import h5py,numpy as np,torch
from .hdf5_data import read_graph_case
from .models import build_model
from .train import tensor_case
from .opensees_adapter import DummyOpenSeesBuilder,load_parameters

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--model",required=True);p.add_argument("--input-hdf5");p.add_argument("--case-id");p.add_argument("--metadata-json");p.add_argument("--torch-threads",type=int,default=1);p.add_argument("--output",default="predicted_first_node_peaks.h5");a=p.parse_args();torch.set_num_threads(a.torch_threads)
    cp=torch.load(a.checkpoint,map_location="cpu");d=cp["dims"];model=build_model(a.model,node_dim=d[0],edge_dim=d[1],global_dim=d[2],motion_dim=d[3],response_dim=6,hidden=cp.get("hidden",96));model.load_state_dict(cp["model"]);model.eval()
    if a.input_hdf5:
        if not a.case_id:p.error("--case-id required with --input-hdf5")
        graph=read_graph_case(a.input_hdf5,a.case_id)
    else:
        if not a.metadata_json:p.error("Provide --input-hdf5 and --case-id")
        raw=DummyOpenSeesBuilder()(load_parameters(a.metadata_json));from .schema import GraphCase;graph=GraphCase(**{k:raw[k] for k in ("global_features","ground_motion","time","node_features","edge_index","edge_features","node_ids","pile_ids")},metadata=raw.get("metadata",{}))
    ids=graph.metadata.get("recorded_node_ids",[])
    if not ids:p.error("Graph input must contain recorded_node_ids metadata")
    node_id=int(ids[0]);matches=np.flatnonzero(np.asarray(graph.node_ids,dtype=np.int64)==node_id)
    if len(matches)!=1:p.error(f"First recorder node {node_id} is not present exactly once")
    with torch.no_grad():
        scaled=model(tensor_case(graph,cp.get("input_normalizer")))[int(matches[0])]
        peaks=(scaled*torch.as_tensor(cp.get("target_scale",[1.]*6),dtype=scaled.dtype)).numpy().astype("f4")
    with h5py.File(a.output,"w") as h:
        h.attrs["format"]="SSI-ML-Predicted-FirstNode-Peaks-v1";h.attrs["target_node_id"]=node_id;h.attrs["response_components"]=json.dumps(["Ux","Uy","Uz","Rx","Ry","Rz"]);h.attrs["definition"]="maximum absolute value of each DOF over response history";h.create_dataset("predicted_peak_abs_response",data=peaks);h.create_dataset("node_id",data=node_id)
    print(f"Predicted six first-node peak values to {a.output}")
if __name__=="__main__":main()
