"""Load previously trained SSI fold checkpoints and combine predictions (no training)."""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import h5py
import numpy as np
from ssi_ml_surrogate_v1.predict_eq_descriptors_first2_v15_gpr import predict


def available_models(models_dir):
    root=Path(models_dir)
    manifest=json.loads((root/'manifest.json').read_text())
    result={}
    for item in manifest['models']:
        result.setdefault(item['name'],[]).append(item)
    for name in result:
        result[name].sort(key=lambda d:int(d['fold'].split('_')[-1]))
    return result


def predict_folds(input_hdf5,case_id,models_dir,model_name,fold=None,verify_hash=True):
    root=Path(models_dir).resolve()
    entries=available_models(root)[model_name]
    if fold is not None:
        entries=[e for e in entries if e['fold']==fold]
    if not entries:
        raise ValueError('No matching checkpoints')
    predictions=[]
    gpr_std=[]
    folds=[]
    with tempfile.TemporaryDirectory() as tmp:
        for index,entry in enumerate(entries):
            checkpoint=(root/entry['file']).resolve()
            if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
                raise ValueError(f'Missing/unsafe checkpoint: {entry["file"]}')
            if verify_hash and hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()!=entry['sha256']:
                raise ValueError(f'Hash mismatch: {entry["file"]}')
            out=Path(tmp)/f'{index}.h5'
            predict(str(input_hdf5),str(case_id),str(checkpoint),str(out))
            with h5py.File(out,'r') as h:
                y=np.asarray(h['predicted_peak_abs_response'][:],dtype=np.float64).reshape(-1)
                if y.shape!=(2,):
                    raise ValueError(f'Expected [Ux,Uy], got {y.shape}')
                if 'predictive_std' in h:
                    gpr_std.append(np.asarray(h['predictive_std'][:],dtype=np.float64).reshape(-1))
            predictions.append(y)
            folds.append(entry['fold'])
    arr=np.stack(predictions)
    result={'model':model_name,'folds':folds,'predictions_m':arr.tolist(),
            'mean_m':arr.mean(axis=0).tolist(),
            'fold_std_m':arr.std(axis=0,ddof=1).tolist() if len(arr)>1 else [0.0,0.0]}
    if gpr_std:
        result['gpr_predictive_std_by_fold_m']=np.stack(gpr_std).tolist()
    return result
