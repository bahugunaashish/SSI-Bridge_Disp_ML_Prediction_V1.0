"""SSI saved-checkpoint inference with Google Drive HDF5 and upload alternatives."""
import json
import tempfile
from pathlib import Path

import h5py
import streamlit as st
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from inference_ensemble import available_models, predict_folds

st.set_page_config(page_title="SSI ML Surrogate", layout="wide")
st.title("SSI peak displacement surrogate")
st.caption("Saved checkpoint inference only; no retraining or OpenSees dynamic analysis.")
root = Path(__file__).resolve().parent / "deployed_models"
if not (root / "manifest.json").is_file():
    st.error("Missing deployed_models/manifest.json and saved checkpoints.")
    st.stop()
try:
    models = available_models(root)
except Exception as exc:
    st.error(f"Could not read manifest: {exc}")
    st.stop()

@st.cache_resource(show_spinner="Downloading the Google Drive HDF5 database (first use only)...")
def download_drive_database(file_id: str, credentials_json: str) -> str:
    """Download one restricted Drive file via service account; cache locally for app lifetime."""
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    session = AuthorizedSession(credentials)
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    # Stream the binary payload to disk, rather than holding ~200 MB in memory.
    response = session.get(url, params={"alt": "media"}, stream=True, timeout=120)
    response.raise_for_status()
    dest = Path(tempfile.mkdtemp(prefix="ssi_drive_hdf5_")) / "input_features.h5"
    try:
        with dest.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
        with h5py.File(dest, "r") as test:
            if not test.keys():
                raise ValueError("Downloaded HDF5 file has no top-level groups")
        return str(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    finally:
        response.close()

mode = st.radio("Prediction mode", ["5-fold ensemble", "Single fold", "Compare all models"], horizontal=True)
model = None
fold = None
if mode != "Compare all models":
    model = st.selectbox("Model", sorted(models))
    if mode == "Single fold":
        fold = st.selectbox("Fold", [e["fold"] for e in models[model]])

source = st.radio("Input source", ["Google Drive database", "Upload HDF5 file"], horizontal=True)
st.info("Use only trusted checkpoint files. Your saved models remain on the server.")

# Uploaded files are stored in a session-local temp directory to survive widget reruns.
path = None
if source == "Google Drive database":
    try:
        file_id = st.secrets["google_drive"]["file_id"]
        credentials_json = st.secrets["google_service_account_json"]
    except (KeyError, FileNotFoundError):
        st.error("Missing Google Drive credentials in Streamlit Secrets. Switch to Upload HDF5 file, or configure Secrets.")
        st.stop()
    if st.button("Connect to Google Drive", type="secondary") or st.session_state.get("drive_connected", False):
        try:
            path = Path(download_drive_database(file_id, credentials_json))
            st.session_state["drive_connected"] = True
            st.success("Google Drive database connected.")
        except Exception as exc:
            st.error(f"Google Drive download failed: {type(exc).__name__}: {exc}")
            st.stop()
else:
    upload = st.file_uploader("Input-only graph HDF5 (.h5 / .hdf5)", type=["h5", "hdf5"])
    if upload is not None:
        upload_signature = (upload.name, upload.size)
        if st.session_state.get("upload_signature") != upload_signature:
            path = Path(tempfile.mkdtemp(prefix="ssi_uploaded_hdf5_")) / "input.h5"
            with path.open("wb") as out:
                while True:
                    chunk = upload.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            st.session_state["upload_path"] = str(path)
            st.session_state["upload_signature"] = upload_signature
        path = Path(st.session_state["upload_path"])

if path is not None:
    try:
        with h5py.File(path, "r") as f:
            cases = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group) and "inputs" in f[k] and "geometry" in f[k])
        if not cases:
            st.error("No compatible simulation groups containing inputs and geometry were found.")
            st.stop()
        st.caption(f"Found {len(cases)} compatible simulations.")
        cid = st.selectbox("Case ID", cases)
        if st.button("Predict", type="primary"):
            names = sorted(models) if mode == "Compare all models" else [model]
            results = {}
            with st.spinner("Loading saved checkpoints and predicting..."):
                for name in names:
                    results[name] = predict_folds(path, cid, root, name, fold=fold if mode == "Single fold" else None)
            for name, result in results.items():
                st.subheader(name.replace("_", " ").upper())
                ux, uy = result["mean_m"]
                a, b = st.columns(2)
                a.metric("Peak |Ux| (mm)", f"{1000*ux:.4f}")
                b.metric("Peak |Uy| (mm)", f"{1000*uy:.4f}")
                st.write("Across-fold SD (mm):", [round(x*1000, 6) for x in result["fold_std_m"]])
                st.dataframe([{"fold": f, "Ux (mm)": 1000*v[0], "Uy (mm)": 1000*v[1]} for f, v in zip(result["folds"], result["predictions_m"])], hide_index=True)
            st.download_button("Download results JSON", json.dumps({"case_id": cid, "results": results}, indent=2), file_name=f"{cid}_predictions.json", mime="application/json")
    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
else:
    st.info("Connect to Google Drive or upload a compatible input-only graph HDF5 file to begin.")
st.caption("Fold disagreement is not a calibrated confidence interval. GPR per-fold predictive standard deviations are included in downloaded JSON.")
