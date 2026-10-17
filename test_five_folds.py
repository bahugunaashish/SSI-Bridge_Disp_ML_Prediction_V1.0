"""Quick local/cluster smoke test of saved models, no retraining."""
import argparse
from inference_ensemble import available_models,predict_folds

p=argparse.ArgumentParser()
p.add_argument('--input-hdf5',required=True)
p.add_argument('--case-id',required=True)
p.add_argument('--models-dir',default='deployed_models')
p.add_argument('--model',default='mlp',help='Model name, or all')
a=p.parse_args()
models=available_models(a.models_dir)
selected=sorted(models) if a.model=='all' else [a.model]
for name in selected:
    print('\nMODEL:',name,flush=True)
    r=predict_folds(a.input_hdf5,a.case_id,a.models_dir,name)
    for fold,value in zip(r['folds'],r['predictions_m']):
        print(f"{fold}: Ux={value[0]:.8f} m; Uy={value[1]:.8f} m")
    print('Mean (mm):', [round(x*1000,5) for x in r['mean_m']])
    print('Fold SD (mm):', [round(x*1000,5) for x in r['fold_std_m']])
