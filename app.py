"""SSI checkpoint-only inference with 5-fold ensembles; no OpenSees imports."""
import json
import tempfile
from pathlib import Path
import h5py
import streamlit as st
from inference_ensemble import available_models,predict_folds

st.set_page_config(page_title='SSI ML Surrogate',layout='wide')
st.title('SSI peak displacement surrogate')
st.caption('Saved checkpoint inference only. Requires an input-only graph HDF5 file; no retraining or OpenSees analysis.')
root=Path(__file__).resolve().parent/'deployed_models'
if not (root/'manifest.json').is_file():
    st.error('Missing deployed_models/manifest.json and saved checkpoints. See README.md.')
    st.stop()
try:
    models=available_models(root)
except Exception as exc:
    st.error(f'Could not read manifest: {exc}')
    st.stop()
mode=st.radio('Prediction mode',['5-fold ensemble','Single fold','Compare all models'],horizontal=True)
model=None
fold=None
if mode!='Compare all models':
    model=st.selectbox('Model',sorted(models))
    if mode=='Single fold':
        fold=st.selectbox('Fold',[e['fold'] for e in models[model]])
st.info('Use only checkpoint files you trust. Checkpoints are server-side; uploaded HDF5 inputs are processed temporarily.')
upload=st.file_uploader('Input-only graph HDF5 (.h5 / .hdf5)',type=['h5','hdf5'])
if upload is not None:
    # Persist uploaded data for a rerun. Streamlit re-executes script for widget selection.
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'input.h5'
        path.write_bytes(upload.getvalue())
        try:
            with h5py.File(path,'r') as f:
                cases=sorted(k for k in f.keys() if isinstance(f[k],h5py.Group) and 'inputs' in f[k] and 'geometry' in f[k])
            if not cases:
                st.error('No compatible case groups containing inputs and geometry were found.')
                st.stop()
            cid=st.selectbox('Case ID',cases)
            if st.button('Predict',type='primary'):
                names=sorted(models) if mode=='Compare all models' else [model]
                results={}
                with st.spinner('Loading saved checkpoints and predicting...'):
                    for name in names:
                        results[name]=predict_folds(path,cid,root,name,fold=fold if mode=='Single fold' else None)
                for name,r in results.items():
                    st.subheader(name.replace('_',' ').upper())
                    ux,uy=r['mean_m']
                    a,b=st.columns(2)
                    a.metric('Peak |Ux| (mm)',f'{1000*ux:.4f}')
                    b.metric('Peak |Uy| (mm)',f'{1000*uy:.4f}')
                    st.write('Across-fold SD (mm):',[round(x*1000,6) for x in r['fold_std_m']])
                    st.dataframe([{'fold':f,'Ux (mm)':1000*v[0],'Uy (mm)':1000*v[1]} for f,v in zip(r['folds'],r['predictions_m'])],hide_index=True)
                st.download_button('Download results JSON',json.dumps({'case_id':cid,'results':results},indent=2),file_name=f'{cid}_predictions.json',mime='application/json')
        except Exception as exc:
            st.error(f'Prediction failed: {type(exc).__name__}: {exc}')
else:
    st.info('Upload a compatible input-only graph HDF5 file to begin.')
st.caption('Fold disagreement is not a calibrated confidence interval. GPR per-fold predictive standard deviations are included in downloaded JSON.')
