# SSI 5-Fold Ensemble Streamlit Deployment

Inference only. **No OpenSeesPy or retraining is required** when a compatible input-only graph HDF5 is supplied. Preparing *new* input graphs from raw engineering parameters using the original extractor does use OpenSeesPy to build geometry, but not to run dynamics.

## On the cluster

1. Export checkpoints as already done: `~/Peak_ML_MetaData/deployed_models/` (35 files + manifest).
2. Copy the files in this package to a GitHub repository directory. Copy the full existing `deployed_models/` directory into that same directory.
3. From the original project root (the parent of `ssi_ml_surrogate_v1`), test **all five MLP folds**:

```bash
python /path/to/ssi_streamlit_deployment/test_five_folds.py \
 --input-hdf5 /scratch/abahugu/TH/MPI_ML/run_output/ml_inputs/first_node_peak_input_features.h5 \
 --case-id simulation_175 \
 --models-dir ~/Peak_ML_MetaData/deployed_models --model mlp
```

This outputs fold predictions, mean and standard deviation, and verifies a no-retraining inference path. To test all 35 checkpoints set `--model all` (may take longer).

## Local UI

From the deployment directory containing `deployed_models/`, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload an input-only graph HDF5 with `simulation_N/inputs`, `simulation_N/geometry`, and compatible feature schema. Choose model and `5-fold ensemble` or `single fold` or `all models`.

## GitHub + Streamlit Community Cloud

- Place `app.py`, `requirements.txt`, `ssi_ml_surrogate_v1/`, and **the entire** `deployed_models/` folder at repository root.
- Check the GitHub upload size limits, and do not publish confidential research models/data in a public repository.
- Commit/push from the local workstation or cluster (if allowed). Connect the repository at https://share.streamlit.io/ with `app.py` as the main file.
- Model files are binary. A `manifest.json` by itself will not let the app predict.
- Cloud memory/CPU limits and Python/package version compatibility can affect graph/GPR inference. Test an actual HDF5 input after deployment. Never upload private or sensitive HDF5 data to a public/shared app.

**Interpretation:** Fold standard deviation measures model-to-model disagreement, not a calibrated interval. Some validation cases may have been part of training for other folds; this smoke test is for functionality, not unbiased accuracy.
