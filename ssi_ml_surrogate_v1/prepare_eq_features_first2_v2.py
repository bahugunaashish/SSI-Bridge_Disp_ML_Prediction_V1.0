"""Build a compact HDF5 with EQ descriptors and first-node Ux/Uy peak labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import h5py
import numpy as np
from .eq_descriptors_v1 import extract_eq_descriptors,GLOBAL_FEATURE_NAMES,WINDOW_FEATURE_NAMES

def run(database, output, windows=24, workers=1):
    source=Path(database); out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    if workers<1: raise ValueError("workers must be positive")
    made=skipped=reused=0
    def write_case(dst,src,cid,features):
        global_eq,seq=features;root=src[cid];targets=root["targets"]
        grp=dst.create_group(cid); grp.create_dataset("static",data=root["inputs"]["global_features"][:],compression="gzip")
        grp.create_dataset("node_features",data=root["inputs"]["node_features"][:],compression="gzip")
        grp.create_dataset("edge_index",data=root["inputs"]["edge_index"][:],compression="gzip")
        grp.create_dataset("edge_features",data=root["inputs"]["edge_features"][:],compression="gzip")
        grp.create_dataset("node_ids",data=root["geometry"]["node_ids"][:])
        grp.create_dataset("eq_global",data=global_eq,compression="gzip");grp.create_dataset("eq_sequence",data=seq,compression="gzip")
        grp.create_dataset("target",data=targets["peak_abs_response"][:2]);grp.create_dataset("mask",data=targets["target_mask"][:2])
        md=json.loads(root.attrs.get("metadata_json","{}"));grp.attrs["recorded_node_ids"]=json.dumps(md.get("recorded_node_ids",[]))
    def finish_one(dst,src,pending):
        cid,future=pending
        try: write_case(dst,src,cid,future.result());return 1,0
        except Exception as e:
            if cid in dst: del dst[cid]
            print(f"SKIP {cid}: {e}",flush=True);return 0,1
    with h5py.File(source,"r") as src,h5py.File(out,"a") as dst:
        if "format" not in dst.attrs:
            dst.attrs["format"]="SSI-ML-EQ-descriptor-First2-v2"
            dst.attrs["feature_definition"]="24 ordered windows x 20 time/frequency features plus global FFT/Prony descriptors"
            dst.attrs["global_eq_feature_names_json"]=json.dumps(GLOBAL_FEATURE_NAMES)
            dst.attrs["window_feature_names_json"]=json.dumps(WINDOW_FEATURE_NAMES)
            dst.attrs["response_components_json"] = json.dumps(["Ux", "Uy"])
            dst.attrs["target_definition"] = "first recorder node max(abs(Ux)), max(abs(Uy)); time column excluded"
        elif dst.attrs["format"]!="SSI-ML-EQ-descriptor-First2-v2":
            raise ValueError(f"Unexpected descriptor database format in {out}")
        pending=[];limit=max(1,workers*2)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for cid in sorted(src.keys()):
                if cid in dst:
                    required={"static","node_features","edge_index","edge_features","node_ids","eq_global","eq_sequence","target","mask"}
                    if required.issubset(dst[cid].keys()): reused+=1;continue
                    del dst[cid]
                try:
                    root=src[cid]
                    if "peak_abs_response" not in root["targets"] or root["targets"]["peak_abs_response"].size < 2: raise ValueError("first-node Ux/Uy peak_abs_response missing")
                    motion=root["inputs"]["ground_motion"][:];time=root["inputs"]["time"][:]
                    pending.append((cid,pool.submit(extract_eq_descriptors,motion,time,windows)))
                    if len(pending)>=limit:
                        n,e=finish_one(dst,src,pending.pop(0));made+=n;skipped+=e
                except Exception as e:
                    skipped+=1;print(f"SKIP {cid}: {e}",flush=True)
            for item in pending:
                n,e=finish_one(dst,src,item);made+=n;skipped+=e
    print(f"Descriptor database: {out}; added={made}, reused existing={reused}, skipped={skipped}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--database",required=True);p.add_argument("--output",required=True);p.add_argument("--windows",type=int,default=24);p.add_argument("--workers",type=int,default=1);a=p.parse_args();run(a.database,a.output,a.windows,a.workers)
if __name__=="__main__":main()
