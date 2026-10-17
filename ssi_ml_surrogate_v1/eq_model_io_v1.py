"""Shared graph-to-static-descriptor packing for training and inference."""
import numpy as np

def first_recorder_index(node_ids, metadata):
    rec=metadata.get("recorded_node_ids",[])
    if not rec: raise ValueError("Input database case has no recorded_node_ids")
    ix=np.flatnonzero(np.asarray(node_ids,dtype=np.int64)==int(rec[0]))
    if len(ix)!=1: raise ValueError("First recorder node was not found uniquely in the input graph")
    return int(ix[0])

def pack_static_features(global_features,node_features,edge_features,node_ids,metadata,eq_global,edge_index=None):
    ix=first_recorder_index(node_ids,metadata)
    nf=np.asarray(node_features,dtype=np.float32);ef=np.asarray(edge_features,dtype=np.float32)
    pool=lambda x: (np.r_[x.mean(0),x.max(0)] if len(x) else np.zeros(x.shape[1]*2,dtype=np.float32)) if x.ndim==2 else np.array([],dtype=np.float32)
    edge_index=np.asarray(edge_index if edge_index is not None else np.empty((2,0)),dtype=np.int64)
    if edge_index.size:
        src,dst=edge_index
        connected=np.unique(np.r_[src[dst==ix],dst[src==ix]])
        connected=connected[connected!=ix]
        local_e=np.flatnonzero((src==ix)|(dst==ix))
        degree=np.bincount(np.r_[src,dst],minlength=len(nf)).astype(np.float32)
        degree_stats=np.asarray([degree[ix],degree.mean(),degree.std(),degree.max()],dtype=np.float32)
    else:
        connected=np.asarray([],dtype=np.int64);local_e=np.asarray([],dtype=np.int64);degree_stats=np.zeros(4,dtype=np.float32)
    neighbor=pool(nf[connected]) if len(connected) else np.zeros(nf.shape[1]*2,dtype=np.float32)
    local_edge=pool(ef[local_e]) if len(local_e) else np.zeros(ef.shape[1]*2,dtype=np.float32)
    return np.r_[global_features,nf[ix],neighbor,pool(nf),local_edge,pool(ef),degree_stats,eq_global].astype(np.float32)
