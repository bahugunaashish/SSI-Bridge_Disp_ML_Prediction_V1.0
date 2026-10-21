"""SSI saved-checkpoint inference with Google Drive HDF5 and upload alternatives."""
import json
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

tab_predict, tab_compare, tab_overfit, tab_case_study, tab_new_prediction, tab_about = st.tabs([
    "Predict a simulation",
    "Actual vs. predicted — 780 cases",
    "Overfitting analysis — train / validation / test",
    "Case study & SHAP analysis",
    "New SSI response prediction",
    "About",
])

with tab_predict:
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


@st.cache_data(show_spinner="Downloading comparison CSV from Google Drive…", ttl=3600)
def load_comparison_csv(file_id: str, credentials_json: str) -> pd.DataFrame:
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    session = AuthorizedSession(credentials)
    response = session.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}",
        params={"alt": "media"}, timeout=120,
    )
    response.raise_for_status()
    df = pd.read_csv(io.BytesIO(response.content))
    required = {"simulation_id", "model", "fold", "actual_Ux_mm", "actual_Uy_mm",
                "predicted_Ux_mm", "predicted_Uy_mm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {sorted(missing)}")
    for col in required - {"simulation_id", "model", "fold"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


ZERO_THRESHOLD_MM = 1e-3


def zero_small_displacements(values):
    """Treat absolute displacements below 0.001 mm as exactly zero."""
    values = np.asarray(values, dtype=float).copy()
    values[np.abs(values) < ZERO_THRESHOLD_MM] = 0.0
    return values


def metric_values(actual, predicted):
    actual = zero_small_displacements(actual)
    predicted = zero_small_displacements(predicted)
    good = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[good], predicted[good]
    if len(actual) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan")
    ss_total = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - np.sum(residual ** 2) / ss_total) if ss_total > 0 else float("nan")
    nonzero = actual != 0.0  # MAPE is undefined for zero actual values
    mape = float(100 * np.mean(np.abs(residual[nonzero] / actual[nonzero]))) if nonzero.any() else float("nan")
    return r2, mape, rmse, residual_std, int(nonzero.sum())


def plot_comparison(data: pd.DataFrame, direction: str, title: str, axis_ranges=None, legend_rmse=False, distinguish_splits=False):
    """Publication-style actual/predicted scatter plot with external legend."""
    actual_col = f"actual_{direction}_mm"
    pred_col = f"predicted_{direction}_mm"
    fig = go.Figure()
    metrics = []
    min_val, max_val = float("inf"), float("-inf")
    palette = ["#1565C0", "#C62828", "#2E7D32", "#7B1FA2",
               "#E65100", "#00838F", "#5D4037"]
    # Each ML model retains its own marker regardless of the selected folds.
    model_names = ["gnn_tcn", "gpr", "graph_operator", "graph_transformer",
                   "meshgraphnet", "mlp", "tcn_lstm"]
    symbols = ["circle-open", "square-open", "diamond-open", "triangle-up-open",
               "triangle-down-open", "cross-open", "x-open"]
    model_colors = {name: palette[i] for i, name in enumerate(model_names)}
    model_symbols = {name: symbols[i] for i, name in enumerate(model_names)}
    legend_count = 0

    for label, subset in data.groupby("series", sort=False):
        actual = zero_small_displacements(subset[actual_col].to_numpy(dtype=float))
        predicted = zero_small_displacements(subset[pred_col].to_numpy(dtype=float))
        valid = np.isfinite(actual) & np.isfinite(predicted)
        actual, predicted = actual[valid], predicted[valid]
        if not len(actual):
            continue
        r2, mape, rmse, residual_std, mape_cases = metric_values(actual, predicted)
        metrics.append({"Series": label, "Cases": len(actual), "MAPE valid cases": mape_cases,
                        "R²": r2, "MAPE (%)": mape, "RMSE (mm)": rmse,
                        "Error SD (mm)": residual_std})
        model = str(subset["model"].iloc[0])
        fold_label = str(subset["fold"].iloc[0]) if " / fold_" in label else "mean"
        marker_symbol = model_symbols[model]
        if distinguish_splits:
            # Dataset split is identifiable by marker shape; model stays color-coded.
            split = str(subset["split"].iloc[0]).lower()
            split_symbols = {"train": "circle-open",
                             "validation": "diamond-open",
                             "test": "square-open"}
            marker_symbol = split_symbols.get(split, marker_symbol)
        color = model_colors[model]
        # Keep existing legends untouched; optionally show RMSE for overfitting plots.
        legend_label = f"{label} | RMSE={rmse:.3f} mm" if legend_rmse else label
        fig.add_trace(go.Scattergl(
            x=actual, y=predicted, mode="markers", name=legend_label,
            customdata=subset.loc[valid, ["simulation_id"]].to_numpy(),
            hovertemplate="Case: %{customdata[0]}<br>Actual: %{x:.4f} mm"
                          "<br>Predicted: %{y:.4f} mm<extra>%{fullData.name}</extra>",
            marker={"symbol": marker_symbol, "size": 8 if fold_label == "mean" else 7,
                    "color": color, "line": {"color": color, "width": 1.2},
                    "opacity": 0.7},
        ))
        min_val = min(min_val, float(actual.min()), float(predicted.min()))
        max_val = max(max_val, float(actual.max()), float(predicted.max()))
        legend_count += 1

    if np.isfinite(min_val) and np.isfinite(max_val):
        pad = max((max_val - min_val) * 0.04, 0.05)
        lower = min(0.0, min_val - pad)
        upper = max_val + pad
        fig.add_trace(go.Scatter(
            x=[lower, upper], y=[lower, upper], mode="lines", name="Ideal (1:1)",
            line={"color": "#505050", "dash": "dash", "width": 1.6},
            hoverinfo="skip",
        ))
        fig.update_xaxes(range=[lower, upper])
        fig.update_yaxes(range=[lower, upper])

    axis_style = dict(
        showgrid=True, gridcolor="#C6C6C6", griddash="dash", gridwidth=1,
        showline=True, mirror=True, linecolor="#222222", linewidth=1.2,
        ticks="outside", ticklen=5, tickcolor="#222222", zeroline=False,
        tickfont={"family": "Courier New, Courier, monospace", "size": 14, "color": "#000000"},
        title_font={"family": "Courier New, Courier, monospace", "size": 17, "color": "#000000"},
        automargin=True,
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.43, "xanchor": "center",
               "font": {"family": "Courier New, Courier, monospace", "size": 20,
                        "color": "#000000"}},
        xaxis_title=f"Actual peak |{direction}| displacement (mm)",
        yaxis_title=f"Predicted peak |{direction}| displacement (mm)",
        font={"family": "Courier New, Courier, monospace", "size": 14, "color": "#000000"},
        plot_bgcolor="white", paper_bgcolor="white",
        legend={"orientation": "v", "x": 1.02, "xanchor": "left",
                "y": 1, "yanchor": "top",
                "font": {"family": "Courier New, Courier, monospace", "size": 12, "color": "#000000"},
                "bgcolor": "rgba(255,255,255,0)",
                "title": {"text": "Model / fold"}},
        height=max(650, min(1150, 180 + legend_count * 26)),
        margin={"l": 95, "r": 260, "t": 85, "b": 95},
        hovermode="closest",
    )
    fig.update_xaxes(**axis_style)
    if axis_ranges is None:
        # Keep the original equal-scale plot in automatic mode.
        fig.update_yaxes(**axis_style, scaleanchor="x", scaleratio=1)
    else:
        # Custom ranges are independent: Plotly scaleanchor would override one range.
        fig.update_yaxes(**axis_style, scaleanchor=False)
        fig.update_xaxes(range=list(axis_ranges[0]), autorange=False)
        fig.update_yaxes(range=list(axis_ranges[1]), autorange=False)
    return fig, pd.DataFrame(metrics)


def comparison_axis_controls(direction: str):
    """Optional independent x/y axis bounds in millimetres for one plot."""
    with st.expander(f"{direction} axis range settings", expanded=False):
        manual = st.checkbox("Set custom axis ranges (mm)", key=f"{direction}_manual_axes")
        if not manual:
            st.caption("Automatic range with equal X and Y scales.")
            return None
        left, right = st.columns(2)
        with left:
            st.markdown("**X axis — actual (mm)**")
            xmin = st.number_input("Minimum X", value=0.0, step=0.5,
                                   format="%.4f", key=f"{direction}_xmin")
            xmax = st.number_input("Maximum X", value=20.0, step=0.5,
                                   format="%.4f", key=f"{direction}_xmax")
        with right:
            st.markdown("**Y axis — predicted (mm)**")
            ymin = st.number_input("Minimum Y", value=0.0, step=0.5,
                                   format="%.4f", key=f"{direction}_ymin")
            ymax = st.number_input("Maximum Y", value=20.0, step=0.5,
                                   format="%.4f", key=f"{direction}_ymax")
        if xmin >= xmax or ymin >= ymax:
            st.error("Axis maximum must be greater than its minimum.")
            return None
        st.caption("Custom X and Y limits are independent; this changes only the view, not the metrics.")
        return ((xmin, xmax), (ymin, ymax))


def plot_metrics_radar(metrics: pd.DataFrame, direction: str, distinguish_splits=False):
    """Compare selected model/fold series using relative 0-1 desirability scores."""
    columns = ["R²", "MAPE (%)", "RMSE (mm)", "Error SD (mm)"]
    labels = ["R² ↑", "MAPE ↓", "RMSE ↓", "Std ↓"]
    # Normalize *per metric* across the currently selected model/fold series.
    # Higher is better for R²; lower is better for the three error metrics.
    scores = pd.DataFrame(index=metrics.index)
    for col in columns:
        values = pd.to_numeric(metrics[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        scaled = np.full(len(values), np.nan, dtype=float)
        if valid.any():
            lo, hi = float(values[valid].min()), float(values[valid].max())
            if np.isclose(hi, lo):
                scaled[valid] = 1.0  # No discrimination when all values tie.
            elif col == "R²":
                scaled[valid] = (values[valid] - lo) / (hi - lo)
            else:
                scaled[valid] = (hi - values[valid]) / (hi - lo)
        scores[col] = scaled

    palette = ["#1565C0", "#C62828", "#2E7D32", "#7B1FA2",
               "#E65100", "#00838F", "#5D4037"]
    names = ["gnn_tcn", "gpr", "graph_operator", "graph_transformer",
             "meshgraphnet", "mlp", "tcn_lstm"]
    shapes = ["circle-open", "square-open", "diamond-open", "triangle-up-open",
              "triangle-down-open", "cross-open", "x-open"]
    colors = dict(zip(names, palette))
    symbols = dict(zip(names, shapes))
    fig = go.Figure()
    for i, row in metrics.iterrows():
        model = str(row["Series"]).split(" / ")[0]
        split = str(row.get("split", "")).lower()
        line_styles = {"train": "dot", "validation": "dash", "test": "solid"}
        style = line_styles.get(split, "solid") if distinguish_splits else "solid"
        series_color = ({"train": "#1565C0", "validation": "#111111",
                         "test": "#D32F2F"}.get(split, "#555555")
                        if distinguish_splits else colors.get(model, "#555555"))
        values = scores.loc[i, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        radial = values.tolist() + [float(values[0])]
        fig.add_trace(go.Scatterpolar(
            r=radial,
            theta=labels + [labels[0]],
            mode="lines+markers",
            name=str(row["Series"]),
            line={"color": colors.get(model, "#555555"), "width": 2.2, "dash": style},
            marker={"symbol": symbols.get(model, "circle-open"), "size": 8,
                    "color": series_color},
            opacity=0.8,
            hovertemplate="%{theta}<br>Relative score: %{r:.3f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        template="plotly_white",
        title={"text": f"Peak |{direction}| — ML model performance (radar)", "x": 0.42,
               "font": {"family": "Courier New, Courier, monospace", "size": 19, "color": "#000000"}},
        font={"family": "Courier New, Courier, monospace", "size": 13, "color": "#000000"},
        polar={"bgcolor": "white",
               "radialaxis": {"range": [0, 1], "tickvals": [0, 0.25, 0.5, 0.75, 1],
                              "gridcolor": "#D0D0D0", "griddash": "dash",
                              "angle": 45, "tickfont": {"size": 12, "color": "#000000"}},
               "angularaxis": {"gridcolor": "#D0D0D0",
                               "tickfont": {"size": 14, "color": "#000000"}}},
        legend={"x": 1.03, "y": 1, "xanchor": "left", "yanchor": "top",
                "title": {"text": "Model / dataset" if distinguish_splits else "Model / fold"},
                "font": {"size": 12, "color": "#000000"}},
        margin={"l": 65, "r": 255, "t": 90, "b": 65},
        height=680,
        paper_bgcolor="white",
    )
    return fig


with tab_compare:
    st.subheader("Actual vs. predicted peak displacements")
    st.caption("Precomputed Clemson-cluster predictions; changing folds/models does not retrain or rerun inference.")
    st.caption("Error SD (mm) is the sample standard deviation of actual minus predicted displacement errors (n − 1 denominator).")
    st.caption("For plots and all metrics, values with |displacement| < 0.001 mm are set to 0. "
               "MAPE excludes cases where actual displacement is zero; 'MAPE valid cases' shows its denominator. "
               "The downloadable CSV retains original, unmodified values.")
    st.warning("Full-training-dataset comparison: metrics here are descriptive, not independent test-set accuracy. "
               "Use held-out or out-of-fold cases for unbiased performance evaluation.")
    try:
        comparison_file_id = st.secrets["google_drive"]["comparison_file_id"]
        comparison_creds = st.secrets["google_service_account_json"]
    except (KeyError, FileNotFoundError):
        st.error("Missing google_drive.comparison_file_id or service-account credentials in Streamlit Secrets.")
    else:
        if st.button("Load comparison database", key="load_comparison") or st.session_state.get("comparison_loaded", False):
            try:
                df = load_comparison_csv(comparison_file_id, comparison_creds)
                st.session_state["comparison_loaded"] = True
                st.caption(f"Loaded {len(df):,} model-fold predictions across {df.simulation_id.nunique():,} simulations.")
                model_options = sorted(df["model"].unique().tolist())
                selected_models = st.multiselect("Models", model_options, default=model_options)
                fold_options = sorted(df["fold"].unique().tolist())
                selected_folds = st.multiselect("Folds (select any combination)", fold_options, default=fold_options)
                display_mode = st.radio("Fold display", ["Mean of selected folds", "Each selected fold", "Both"], horizontal=True)
                if not selected_models or not selected_folds:
                    st.info("Select at least one model and one fold to view plots.")
                else:
                    filtered = df[df["model"].isin(selected_models) & df["fold"].isin(selected_folds)].copy()
                    series_parts = []
                    if display_mode in ("Each selected fold", "Both"):
                        indiv = filtered.copy()
                        indiv["series"] = indiv["model"] + " / " + indiv["fold"]
                        series_parts.append(indiv)
                    if display_mode in ("Mean of selected folds", "Both"):
                        averaged = filtered.groupby(["simulation_id", "model"], as_index=False).agg(
                            actual_Ux_mm=("actual_Ux_mm", "first"),
                            actual_Uy_mm=("actual_Uy_mm", "first"),
                            predicted_Ux_mm=("predicted_Ux_mm", "mean"),
                            predicted_Uy_mm=("predicted_Uy_mm", "mean"),
                        )
                        averaged["series"] = averaged["model"] + " / mean"
                        series_parts.append(averaged)
                    chart_df = pd.concat(series_parts, ignore_index=True)
                    x_tab, y_tab = st.tabs(["Peak |Ux|", "Peak |Uy|"])
                    with x_tab:
                        x_ranges = comparison_axis_controls("Ux")
                        fig_x, metrics_x = plot_comparison(
                            chart_df, "Ux", "Peak |Ux|: actual vs. predicted", axis_ranges=x_ranges)
                        st.plotly_chart(fig_x, use_container_width=True)
                        st.dataframe(metrics_x, hide_index=True)
                        st.subheader("Model performance spider plot — Ux")
                        st.caption("Relative scores (0–1) within the displayed model/fold selection; "
                                   "higher is better on every spoke. R² is maximized; MAPE, RMSE, "
                                   "and error SD are minimized. Tied metrics score 1 for all series. "
                                   "Use the table above for actual values and units.")
                        st.plotly_chart(plot_metrics_radar(metrics_x, "Ux"), use_container_width=True)
                    with y_tab:
                        y_ranges = comparison_axis_controls("Uy")
                        fig_y, metrics_y = plot_comparison(
                            chart_df, "Uy", "Peak |Uy|: actual vs. predicted", axis_ranges=y_ranges)
                        st.plotly_chart(fig_y, use_container_width=True)
                        st.dataframe(metrics_y, hide_index=True)
                        st.subheader("Model performance spider plot — Uy")
                        st.caption("Relative scores (0–1) within the displayed model/fold selection; "
                                   "higher is better on every spoke. R² is maximized; MAPE, RMSE, "
                                   "and error SD are minimized. Tied metrics score 1 for all series. "
                                   "Use the table above for actual values and units.")
                        st.plotly_chart(plot_metrics_radar(metrics_y, "Uy"), use_container_width=True)
                    st.download_button("Download selected predictions (CSV)",
                                       chart_df.to_csv(index=False).encode("utf-8"),
                                       file_name="ssi_selected_predictions.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Could not load or plot comparison database: {type(exc).__name__}: {exc}")

# Independent overfitting analysis: uses a separate, split-labeled CSV.
# The existing prediction and all-data comparison tabs are deliberately unchanged.
OVERFITTING_DRIVE_FILE_ID = "1GwaPgm5lVMobPWNKRhKM3TT9w0AiRyMz"



def overfit_axis_controls(direction):
    """Independent manual X/Y axis controls for the overfitting tab only."""
    with st.expander(f"{direction} axis range settings", expanded=False):
        manual = st.checkbox(
            "Set custom axis ranges (mm)",
            key=f"overfit_{direction}_manual_axes")
        if not manual:
            st.caption("Automatic range with equal X and Y scales.")
            return None
        xcol, ycol = st.columns(2)
        with xcol:
            st.markdown("**X axis — actual (mm)**")
            xmin = st.number_input(
                "Minimum X", value=0.0, step=0.5, format="%.4f",
                key=f"overfit_{direction}_xmin")
            xmax = st.number_input(
                "Maximum X", value=20.0, step=0.5, format="%.4f",
                key=f"overfit_{direction}_xmax")
        with ycol:
            st.markdown("**Y axis — predicted (mm)**")
            ymin = st.number_input(
                "Minimum Y", value=0.0, step=0.5, format="%.4f",
                key=f"overfit_{direction}_ymin")
            ymax = st.number_input(
                "Maximum Y", value=20.0, step=0.5, format="%.4f",
                key=f"overfit_{direction}_ymax")
        if xmin >= xmax or ymin >= ymax:
            st.error("Axis maximum must exceed its minimum.")
            st.stop()
        st.caption("Axis limits only change the view; they do not alter performance metrics.")
        return ((xmin, xmax), (ymin, ymax))


def overfit_metrics_table(data, direction, group_columns):
    """Raw, unnormalized performance metrics for each requested group."""
    rows = []
    for group_key, group in data.groupby(group_columns, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        a = group[f"actual_{direction}_mm"].to_numpy(dtype=float)
        p = group[f"predicted_{direction}_mm"].to_numpy(dtype=float)
        r2, mape, rmse, sd, n_mape = metric_values(a, p)
        row = dict(zip(group_columns, group_key))
        row.update({"N": len(group), "MAPE valid cases": n_mape,
                    "R²": r2, "MAPE (%)": mape, "RMSE (mm)": rmse,
                    "Error SD (mm)": sd})
        rows.append(row)
    return pd.DataFrame(rows)


def overfit_fold_subplots(data, direction, fold_order, axis_ranges=None):
    """One actual-vs-predicted panel per selected fold; all models/splits in each."""
    import math
    splits = {"train": "circle-open", "validation": "diamond-open",
              "test": "square-open"}
    # Consistent dataset colors across every fold and every ML model.
    colors = {"train": "#1565C0", "validation": "#111111", "test": "#D32F2F"}
    # Stack folds vertically so enlarged labels and individual legends never
    # encroach on an adjacent fold's plotting area.
    ncols = 1
    nrows = len(fold_order)
    fig = make_subplots(
        rows=nrows, cols=ncols,
        horizontal_spacing=0.0,
        vertical_spacing=0.065 if nrows > 1 else 0.0,
    )
    xcol, ycol = f"actual_{direction}_mm", f"predicted_{direction}_mm"
    global_values = []
    for panel, fold in enumerate(fold_order):
        row, col = divmod(panel, ncols)
        row += 1
        col += 1
        portion = data.loc[data["fold"].eq(fold)]
        for (model, split), points in portion.groupby(["model", "split"], sort=True):
            actual = zero_small_displacements(points[xcol].to_numpy(dtype=float))
            predicted = zero_small_displacements(points[ycol].to_numpy(dtype=float))
            valid = np.isfinite(actual) & np.isfinite(predicted)
            if not valid.any():
                continue
            actual, predicted = actual[valid], predicted[valid]
            global_values.extend((actual.min(), actual.max(), predicted.min(), predicted.max()))
            series_name = f"Fold {str(fold).replace('fold_', '')} / {split.capitalize()}"
            legend_name = "legend" if panel == 0 else f"legend{panel + 1}"
            fig.add_trace(go.Scattergl(
                x=actual, y=predicted, mode="markers",
                name=series_name, legendgroup=series_name,
                legend=legend_name, showlegend=False,
                marker={
                    "symbol": splits[split], "color": colors.get(split, "#444444"),
                    "size": 10.5, "opacity": 0.7,
                    "line": {"color": colors.get(split, "#444444"), "width": 1.1},
                },
                customdata=points.loc[valid, "simulation_id"].astype(str).to_numpy(),
                hovertemplate=(
                    f"Fold {fold}<br>{model} / {split}<br>"
                    "Case %{customdata}<br>Actual %{x:.4f} mm"
                    "<br>Predicted %{y:.4f} mm<extra></extra>"
                ),
            ), row=row, col=col)
        fig.update_xaxes(title_text=f"Actual peak |{direction}| (mm)", row=row, col=col)
        fig.update_yaxes(title_text=f"Predicted peak |{direction}| (mm)", row=row, col=col)
    if global_values:
        lo, hi = min(global_values), max(global_values)
        pad = max((hi - lo) * 0.04, 0.05)
        default_range = (min(0., lo - pad), hi + pad)
        if default_range[1] <= default_range[0]:
            default_range = (default_range[0], default_range[0] + 1.)
        x_range, y_range = (default_range, default_range) if axis_ranges is None else axis_ranges
        for panel, fold in enumerate(fold_order):
            row, col = divmod(panel, ncols)
            row += 1
            col += 1
            fig.add_trace(go.Scatter(
                x=[min(x_range[0], y_range[0]), max(x_range[1], y_range[1])],
                y=[min(x_range[0], y_range[0]), max(x_range[1], y_range[1])],
                mode="lines", line={"color": "#555555", "dash": "dash", "width": 1.4},
                showlegend=False, legendgroup=f"ideal_{panel}",
                legend="legend" if panel == 0 else f"legend{panel + 1}",
                name="Ideal (1:1)",
                hoverinfo="skip",
            ), row=row, col=col)
            fig.update_xaxes(range=list(x_range), row=row, col=col)
            fig.update_yaxes(range=list(y_range), row=row, col=col)
    axis_style = dict(
        showgrid=True, gridcolor="#C6C6C6", griddash="dash", gridwidth=1,
        showline=True, mirror=True, linecolor="#222222", linewidth=1.2,
        ticks="outside", ticklen=6, zeroline=False,
        tickfont={"family": "Courier New, Courier, monospace", "size": 17, "color": "#000000"},
        title_font={"family": "Courier New, Courier, monospace", "size": 18, "color": "#000000"},
        automargin=True,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_layout(
        template="plotly_white", height=max(620, nrows * 510 + 110),
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Courier New, Courier, monospace", "size": 18, "color": "#000000"},
        title={"text": f"Peak |{direction}| — individual cross-validation folds",
               "x": 0.40, "font": {"size": 30, "color": "#000000"}},
        margin={"l": 105, "r": 45, "t": 100, "b": 85},
        showlegend=False,
        hovermode="closest",
    )
    # Constrain each data panel to a square, rather than a stretched rectangle.
    # Keep X/Y plotting ranges and fitted diagonal consistent for every fold.
    plot_limits = (x_range, y_range) if global_values else ((0.0, 1.0), (0.0, 1.0))
    x_span = max(float(plot_limits[0][1] - plot_limits[0][0]), 1e-12)
    y_span = max(float(plot_limits[1][1] - plot_limits[1][0]), 1e-12)
    for panel, fold in enumerate(fold_order):
        row, col = divmod(panel, ncols)
        row += 1
        col += 1
        xaxis_ref = "x" if panel == 0 else f"x{panel + 1}"
        fig.update_yaxes(
            scaleanchor=xaxis_ref, scaleratio=x_span / y_span,
            constrain="domain", row=row, col=col)
        fig.update_xaxes(constrain="domain", row=row, col=col)
        # Legend-like annotation uses the actual subplot axes, not the
        # surrounding figure. It stays inside the top-left of every panel.
        available = set(data.loc[data["fold"].eq(fold), "split"].astype(str))
        legend_lines = [f"<b>Fold {str(fold).replace('fold_', '')}</b>"]
        for split, color, label in (
                ("train", "#1565C0", "Training (circle)"),
                ("validation", "#111111", "Validation (diamond)"),
                ("test", "#D32F2F", "Testing (square)")):
            if split in available:
                legend_lines.append(
                    f'<span style="color:{color}">{label}</span>')
        legend_lines.append("Ideal (1:1) - dashed")
        suffix = "" if panel == 0 else str(panel + 1)
        fig.add_annotation(
            x=0.025, y=0.975,
            xref=f"x{suffix} domain", yref=f"y{suffix} domain",
            xanchor="left", yanchor="top",
            text="<br>".join(legend_lines), showarrow=False,
            align="left", bgcolor="rgba(255,255,255,0.91)",
            bordercolor="#C6C6C6", borderwidth=1, borderpad=5,
            font={"family": "Courier New, Courier, monospace",
                  "size": 15, "color": "#000000"},
        )
    return fig


def overfit_plot_data(data, fold_detail):
    """Build trace labels without ever averaging across train/validation/test."""
    result = data.copy()
    if fold_detail:
        result["series"] = (result["model"] + " / " + result["fold"] +
                            " / " + result["split"])
    else:
        result["series"] = result["model"] + " / " + result["split"]
    return result


with tab_overfit:
    st.subheader("Overfitting analysis — training, validation, and held-out test")
    st.caption("This tab uses the separately labeled cross-validation CSV; the first two tabs "
               "and their Google Drive files are unchanged.")
    st.caption("All metrics and plot values use |U| < 0.001 mm → 0 mm for both actual and "
               "predicted Ux/Uy. MAPE excludes adjusted zero actual values. "
               "Error SD is sample SD of prediction residuals.")
    st.warning("Validation data were used for model tuning/early stopping. "
               "Held-out test data are the more appropriate estimate of generalization. "
               "When pooling folds, training and validation cases can repeat across folds; "
               "pooled metrics weight each fold-case prediction, not each unique simulation.")
    try:
        overfit_creds = st.secrets["google_service_account_json"]
        overfit_id = st.secrets["google_drive"].get(
            "overfitting_file_id", OVERFITTING_DRIVE_FILE_ID)
    except (KeyError, FileNotFoundError):
        st.error("Google service-account credentials are missing from Streamlit Secrets.")
    else:
        if (st.button("Load training / validation / test data", key="load_overfit")
                or st.session_state.get("overfit_loaded", False)):
            try:
                of = load_comparison_csv(overfit_id, overfit_creds).copy()
                if "split" not in of.columns:
                    raise ValueError("Expected a split column in the new CSV.")
                of["split"] = of["split"].astype(str).str.lower().str.strip()
                expected_splits = {"train", "validation", "test"}
                if not set(of["split"].unique()).issubset(expected_splits):
                    raise ValueError(f"Unexpected split labels: {sorted(of['split'].unique())}")
                keys = ["simulation_id", "model", "fold"]
                if of.duplicated(keys).any():
                    raise ValueError("Duplicate model/fold/simulation rows in CSV.")
                st.session_state["overfit_loaded"] = True
                st.success(f"Loaded {len(of):,} predictions from {of.simulation_id.nunique():,} simulations.")
                breakdown = of.groupby("split", sort=False).size().rename("Predictions").reset_index()
                st.dataframe(breakdown, hide_index=True)
                c_models, c_folds = st.columns(2)
                with c_models:
                    chosen_models = st.multiselect(
                        "Models — overfitting tab",
                        sorted(of["model"].unique()), default=sorted(of["model"].unique()),
                        key="overfit_models")
                with c_folds:
                    chosen_folds = st.multiselect(
                        "Folds — overfitting tab",
                        sorted(of["fold"].unique()), default=sorted(of["fold"].unique()),
                        key="overfit_folds")
                selected_splits = st.multiselect(
                    "Datasets to compare",
                    ["train", "validation", "test"],
                    default=["train", "validation", "test"],
                    key="overfit_splits")
                st.caption("Each selected model has its own expandable plot containing separate fold subplots. Use these selectors to hide or restore models, folds, and datasets.")
                if not chosen_models or not chosen_folds or not selected_splits:
                    st.info("Select at least one model, fold, and dataset.")
                else:
                    selected = of[
                        of["model"].isin(chosen_models)
                        & of["fold"].isin(chosen_folds)
                        & of["split"].isin(selected_splits)
                    ].copy()
                    ux_view, uy_view = st.tabs(["Training / validation / test — Ux",
                                                "Training / validation / test — Uy"])
                    for direction, area in [("Ux", ux_view), ("Uy", uy_view)]:
                        with area:
                            st.markdown(f"### Peak |{direction}| performance")
                            st.caption(
                                "One independent plot per selected ML model, with a "
                                "separate subplot for every selected fold (up to five). "
                                "Use the model/fold selectors above to add or remove "
                                "plots and fold panels.")
                            st.caption("Training: blue ○ circle; validation: black ◇ diamond; "
                                       "testing: red □ square. Colors and shapes identify datasets "
                                       "consistently across all ML models and folds. The legend "
                                       "identifies each model/dataset. "
                                       "RMSE, R², MAPE, and Error SD are in the tables below.")
                            custom_ranges = overfit_axis_controls(direction)
                            for model_name in chosen_models:
                                model_rows = selected.loc[selected["model"].eq(model_name)]
                                if model_rows.empty:
                                    continue
                                with st.expander(
                                        f"{model_name} — {direction} | selected fold panels",
                                        expanded=True):
                                    st.caption(
                                        f"Showing {len(chosen_folds)} fold subplot(s) "
                                        "with the selected training/validation/testing "
                                        "datasets for this model only.")
                                    plot = overfit_fold_subplots(
                                        model_rows, direction, chosen_folds,
                                        axis_ranges=custom_ranges)
                                    plot.update_layout(
                                        title={
                                            "text": f"{model_name} — peak |{direction}| "
                                                    "by cross-validation fold",
                                            "x": 0.40,
                                            "font": {"size": 20},
                                        })
                                    st.plotly_chart(
                                        plot, use_container_width=True,
                                        key=f"overfit_plot_{direction}_{model_name}")
                            st.markdown("**Metrics per model, fold, and dataset**")
                            per_fold = overfit_metrics_table(
                                selected, direction, ["model", "fold", "split"])
                            st.dataframe(per_fold, hide_index=True)
                            st.markdown("**Pooled selected-fold metrics by model and dataset**")
                            pooled = overfit_metrics_table(
                                selected, direction, ["model", "split"])
                            st.dataframe(pooled, hide_index=True)
                            if not pooled.empty:
                                radar_input = pooled.copy()
                                radar_input["Series"] = (radar_input["model"] +
                                                         " / " + radar_input["split"])
                                st.markdown("**Four-metric spider plot (relative 0–1 scores)**")
                                st.caption("Higher relative scores are better. Training: blue/dotted; "
                                           "validation: black/dashed; testing: red/solid. "
                                           "Model marker shapes remain distinct. "
                                           "Actual metrics are in the tables above.")
                                st.plotly_chart(
                                    plot_metrics_radar(radar_input, direction, distinguish_splits=True),
                                    use_container_width=True,
                                    key=f"overfit_radar_{direction}")
                            if {"train", "test"}.issubset(set(selected["split"])):
                                pivot = per_fold.pivot_table(
                                    index=["model", "fold"], columns="split",
                                    values=["R²", "RMSE (mm)", "Error SD (mm)"],
                                    aggfunc="first")
                                gap = pd.DataFrame(index=pivot.index)
                                for metric in ["R²", "RMSE (mm)", "Error SD (mm)"]:
                                    if (metric, "train") in pivot.columns and (metric, "test") in pivot.columns:
                                        sign = -1 if metric == "R²" else 1
                                        gap[f"Test − Train {metric}"] = (
                                            sign * (pivot[(metric, "test")] -
                                                    pivot[(metric, "train")]))
                                if not gap.empty:
                                    st.markdown("**Overfitting gap: held-out test versus training**")
                                    st.caption("Positive gap: R² decreased, or RMSE/Error SD "
                                               "increased on test relative to training.")
                                    st.dataframe(gap.reset_index(), hide_index=True)
                            st.download_button(
                                f"Download {direction} per-fold metrics CSV",
                                per_fold.to_csv(index=False).encode("utf-8"),
                                file_name=f"ssi_overfit_{direction}_metrics.csv",
                                mime="text/csv", key=f"overfit_download_{direction}")
                    st.download_button(
                        "Download selected split-labeled predictions CSV",
                        selected.to_csv(index=False).encode("utf-8"),
                        file_name="ssi_train_validation_test_selected.csv",
                        mime="text/csv", key="overfit_download_predictions")
            except Exception as exc:
                st.error(f"Could not load overfitting analysis data: "
                         f"{type(exc).__name__}: {exc}")
                st.caption("If Google Drive returns HTTP 404, verify the new CSV is shared "
                           "with the same service-account email as the original HDF5 file.")


SHAP_SAMPLES_DRIVE_ID = "1lU6dI-yHVIqDOwUlcg4ayVRSg6ZXO9fE"
CASE_STUDY_FONT = "Courier New, Courier, monospace"
CASE_STUDY_COLORS = {
    "gnn_tcn": "#1565C0", "gpr": "#C62828",
    "graph_operator": "#2E7D32", "graph_transformer": "#7B1FA2",
    "meshgraphnet": "#E65100", "mlp": "#00838F",
    "tcn_lstm": "#5D4037",
}


@st.cache_data(show_spinner="Downloading SHAP samples from Google Drive...", ttl=3600)
def load_shap_samples(file_id, credentials_json):
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    session = AuthorizedSession(credentials)
    response = session.get(
        f"https://www.googleapis.com/drive/v3/files/{file_id}",
        params={"alt": "media"}, timeout=120)
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content))
    required = {
        "fold", "simulation", "model", "output", "feature",
        "input_feature_mean", "input_feature_std", "input_feature_min",
        "input_feature_max", "shapley_value",
    }
    if missing := (required - set(frame.columns)):
        raise ValueError(f"SHAP samples missing columns: {sorted(missing)}")
    frame = frame.rename(columns={"simulation": "simulation_id"})
    frame["fold"] = "fold_" + frame["fold"].astype(str).str.replace(
        r"^fold_", "", regex=True)
    frame["output"] = frame["output"].astype(str)
    for name in ("input_feature_mean", "input_feature_std",
                 "input_feature_min", "input_feature_max", "shapley_value"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


CASE_STUDY_SYMBOLS = {
    "gnn_tcn": "circle", "gpr": "square",
    "graph_operator": "diamond", "graph_transformer": "triangle-up",
    "meshgraphnet": "triangle-down", "mlp": "cross", "tcn_lstm": "x",
}


def case_axis_controls(direction, kind, x_label, y_label):
    """Independent X/Y manual limits, scoped to each case-study plot."""
    scope = f"shap_{direction}_{kind}"
    with st.expander(f"{direction} — {kind} axis range settings", expanded=False):
        manual = st.checkbox("Set custom X and Y axis ranges",
                             key=f"{scope}_manual")
        if not manual:
            st.caption("Automatically determined axis limits.")
            return None
        xc, yc = st.columns(2)
        with xc:
            st.markdown(f"**X axis — {x_label}**")
            xmin = st.number_input("Minimum X", value=0.0, step=0.5,
                                   format="%.5f", key=f"{scope}_xmin")
            xmax = st.number_input("Maximum X", value=20.0, step=0.5,
                                   format="%.5f", key=f"{scope}_xmax")
        with yc:
            st.markdown(f"**Y axis — {y_label}**")
            ymin = st.number_input("Minimum Y", value=0.0, step=0.5,
                                   format="%.5f", key=f"{scope}_ymin")
            ymax = st.number_input("Maximum Y", value=20.0, step=0.5,
                                   format="%.5f", key=f"{scope}_ymax")
        if xmin >= xmax or ymin >= ymax:
            st.error("The axis maximum must be greater than the minimum.")
            return None
        st.caption("Manual limits change only the plot view, not calculations.")
        return ((xmin, xmax), (ymin, ymax))


def case_plot_style(fig, x_title, y_title, title, axis_ranges=None):
    """Reuse comparison-tab publication styling, with optional custom limits."""
    axis_style = dict(
        showgrid=True, gridcolor="#C6C6C6", griddash="dash", gridwidth=1,
        showline=True, mirror=True, linecolor="#222222", linewidth=1.2,
        ticks="outside", ticklen=5, tickcolor="#222222", zeroline=False,
        tickfont={"family": CASE_STUDY_FONT, "size": 14, "color": "#000000"},
        title_font={"family": CASE_STUDY_FONT, "size": 17, "color": "#000000"},
        automargin=True,
    )
    fig.update_layout(
        template="plotly_white", height=650, paper_bgcolor="white",
        plot_bgcolor="white",
        title={"text": title, "x": 0.43, "xanchor": "center",
               "font": {"family": CASE_STUDY_FONT, "color": "#000000", "size": 20}},
        font={"family": CASE_STUDY_FONT, "color": "#000000", "size": 14},
        xaxis_title=x_title, yaxis_title=y_title,
        legend={"orientation": "v", "x": 1.02, "y": 1,
                "xanchor": "left", "yanchor": "top",
                "font": {"family": CASE_STUDY_FONT, "color": "#000000", "size": 12},
                "bgcolor": "rgba(255,255,255,0)"},
        margin={"l": 95, "r": 260, "t": 85, "b": 95},
        hovermode="closest")
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    if axis_ranges is not None:
        fig.update_xaxes(range=list(axis_ranges[0]), autorange=False)
        fig.update_yaxes(range=list(axis_ranges[1]), autorange=False)
    return fig


def add_shap_percentages(data):
    """Signed percent of total absolute feature contribution, per explained case."""
    result = data.copy()
    keys = ["simulation_id", "model", "fold", "output"]
    denominator = result.groupby(keys)["shapley_value"].transform(
        lambda values: values.abs().sum(min_count=1))
    result["shapley_percent"] = np.where(
        denominator > 0, 100.0 * result["shapley_value"] / denominator, np.nan)
    return result


def case_comparison_scatter(data, feature, direction, stat, kind, axis_ranges=None):
    """Compare selected simulations using model and fold marker styling."""
    fig = go.Figure()
    subset = data.loc[data["feature"].eq(feature) & data["output"].eq(direction)]
    for (model, fold), rows in subset.groupby(["model", "fold"], sort=True):
        fold_number = int(str(fold).replace("fold_", ""))
        fold_size = 7 + 2 * fold_number
        color = CASE_STUDY_COLORS.get(model, "#333333")
        shape = CASE_STUDY_SYMBOLS.get(model, "circle")
        x = rows[stat].to_numpy(dtype=float)
        case_meta = rows[["simulation_id", "split"]].astype(str).to_numpy()
        if kind == "displacement":
            # Fold shapes are distinct; model identity is retained by color.
            fold_shapes = {1: "circle", 2: "square", 3: "diamond",
                           4: "triangle-up", 5: "triangle-down"}
            fold_shape = fold_shapes.get(fold_number, "circle")
            traces = [("Actual", f"actual_{direction}_mm", fold_shape + "-open"),
                      ("Predicted", f"predicted_{direction}_mm", fold_shape)]
            for title, column, marker in traces:
                y = zero_small_displacements(rows[column].to_numpy(dtype=float))
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers", name=f"{model} / {fold} / {title}",
                    marker={"symbol": marker, "color": color, "size": fold_size,
                            "line": {"color": color, "width": 1.4}},
                    customdata=case_meta,
                    hovertemplate="Case %{customdata[0]}<br>Split %{customdata[1]}"
                                  "<br>Input %{x:.5g}<br>Displacement %{y:.5g} mm"
                                  "<extra>%{fullData.name}</extra>"))
            ylabel = f"Peak |{direction}| displacement (mm)"
            title = f"Input feature vs actual/predicted |{direction}|"
        else:
            y = rows["shapley_percent"].to_numpy(dtype=float)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{model} / {fold}",
                marker={"symbol": shape + "-open", "color": color, "size": fold_size,
                        "line": {"color": color, "width": 1.4}},
                customdata=case_meta,
                hovertemplate="Case %{customdata[0]}<br>Split %{customdata[1]}"
                              "<br>Input %{x:.5g}<br>Relative SHAP %{y:.3f}%"
                              "<extra>%{fullData.name}</extra>"))
            ylabel = f"Relative Shapley contribution to |{direction}| (%)"
            title = f"Input feature vs SHAP contribution — |{direction}|"
    if kind != "displacement":
        fig.add_hline(y=0, line_dash="dash", line_color="#505050")
    return case_plot_style(
        fig, f"{feature} ({stat.replace('input_feature_', '')})",
        ylabel, title, axis_ranges=axis_ranges)


def case_comparison_importance(data, direction):
    """Mean absolute Shapley contributions for selected cases, by model."""
    df = data.loc[data["output"].eq(direction)].copy()
    df["abs_shap_pct"] = df["shapley_percent"].abs()
    ranking = df.groupby("feature")["abs_shap_pct"].mean().nlargest(15).index.tolist()
    summary = df.loc[df["feature"].isin(ranking)].groupby(
        ["model", "feature"])["abs_shap_pct"].mean()
    fig = go.Figure()
    for model in sorted(df["model"].unique()):
        values = [float(summary.get((model, feature), 0)) for feature in ranking]
        fig.add_trace(go.Bar(
            x=values, y=ranking, orientation="h", name=model,
            marker_color=CASE_STUDY_COLORS.get(model, "#333333")))
    fig.update_layout(barmode="group", height=750)
    fig.update_yaxes(autorange="reversed")
    return case_plot_style(
        fig, "Mean absolute relative Shapley contribution (%)", "Input feature",
        f"Top 15 features — selected cases — {direction}")


with tab_case_study:
    st.subheader("Case study — SHAP and peak displacement comparisons")
    st.caption("Compare multiple models, folds and simulations together. Only sampled "
               "held-out simulations have saved SHAP values. SHAP values are approximate "
               "and normalized to signed percentages of total absolute contribution "
               "per simulation/model/fold/output. This is NOT a percent of predicted "
               "displacement; node, edge and earthquake inputs use channel summaries.")
    try:
        shap_creds = st.secrets["google_service_account_json"]
        shap_id = st.secrets["google_drive"].get(
            "shap_samples_file_id", SHAP_SAMPLES_DRIVE_ID)
        prediction_id = st.secrets["google_drive"].get(
            "overfitting_file_id", OVERFITTING_DRIVE_FILE_ID)
    except (KeyError, FileNotFoundError):
        st.error("Missing Google service-account credentials.")
    else:
        if (st.button("Load case-study data", key="load_shap_case")
                or st.session_state.get("shap_case_loaded", False)):
            try:
                shap_data = load_shap_samples(shap_id, shap_creds)
                predictions = load_comparison_csv(prediction_id, shap_creds)
                if "split" not in predictions.columns:
                    raise ValueError("Split column missing from prediction CSV.")
                join = ["simulation_id", "model", "fold"]
                if predictions.duplicated(join).any():
                    raise ValueError("Duplicate prediction records.")
                columns = join + ["split", "actual_Ux_mm", "predicted_Ux_mm",
                                  "actual_Uy_mm", "predicted_Uy_mm"]
                data = shap_data.merge(
                    predictions[columns], on=join, how="inner",
                    validate="many_to_one")
                if data.empty:
                    raise ValueError("SHAP and prediction datasets have no matching cases.")
                data = add_shap_percentages(data)
                st.session_state["shap_case_loaded"] = True
                st.caption(f"{data.simulation_id.nunique()} explained simulations match "
                           "the prediction database.")
                left, right = st.columns(2)
                model_options = sorted(data["model"].unique())
                fold_options = sorted(data["fold"].unique())
                chosen_models = left.multiselect(
                    "Models", model_options, default=model_options,
                    key="shap_compare_models")
                chosen_folds = right.multiselect(
                    "Folds", fold_options, default=fold_options,
                    key="shap_compare_folds")
                filtered = data.loc[
                    data["model"].isin(chosen_models) &
                    data["fold"].isin(chosen_folds)].copy()
                if filtered.empty:
                    st.info("Select at least one model and fold.")
                else:
                    feature = st.selectbox(
                        "Input feature", sorted(filtered["feature"].unique()),
                        key="shap_compare_feature")
                    available_cases = sorted(filtered.loc[
                        filtered["feature"].eq(feature), "simulation_id"].unique())
                    chosen_cases = st.multiselect(
                        "Simulations (select one, several, or all)",
                        available_cases, default=available_cases,
                        key="shap_compare_cases")
                    stat = st.selectbox(
                        "Input feature summary (X axis)",
                        ["mean", "min", "max", "std"], key="shap_case_stat")
                    chosen_data = filtered.loc[
                        filtered["simulation_id"].isin(chosen_cases)].copy()
                    if chosen_data.empty:
                        st.info("Select at least one simulation.")
                    else:
                        st.caption("In displacement plots, colors identify models and shapes "
                                   "identify folds; open markers = actual, filled = predicted. "
                                   "SHAP plots retain their existing marker scheme.")
                        overview = chosen_data[
                            columns].drop_duplicates().copy()
                        for col in ("actual_Ux_mm", "predicted_Ux_mm",
                                    "actual_Uy_mm", "predicted_Uy_mm"):
                            overview[col] = zero_small_displacements(overview[col])
                        st.dataframe(overview, hide_index=True)
                        ux_tab, uy_tab = st.tabs(["Case study Ux", "Case study Uy"])
                        for direction, panel in (("Ux", ux_tab), ("Uy", uy_tab)):
                            with panel:
                                input_label = f"{feature} ({stat})"
                                limits = case_axis_controls(
                                    direction, "displacement", input_label,
                                    f"Peak |{direction}| displacement (mm)")
                                st.plotly_chart(case_comparison_scatter(
                                    chosen_data, feature, direction,
                                    "input_feature_" + stat, "displacement", limits),
                                    use_container_width=True,
                                    key=f"shap_compare_disp_{direction}")
                                limits = case_axis_controls(
                                    direction, "contribution", input_label,
                                    "Relative Shapley contribution (%)")
                                st.plotly_chart(case_comparison_scatter(
                                    chosen_data, feature, direction,
                                    "input_feature_" + stat, "shap", limits),
                                    use_container_width=True,
                                    key=f"shap_compare_phi_{direction}")
                                st.caption("Feature importance uses mean absolute relative SHAP (%) "
                                           "across the selected cases and folds.")
                                st.plotly_chart(
                                    case_comparison_importance(chosen_data, direction),
                                    use_container_width=True,
                                    key=f"shap_compare_importance_{direction}")
                        st.download_button(
                            "Download selected SHAP cases with predictions",
                            chosen_data.to_csv(index=False).encode("utf-8"),
                            file_name="selected_shap_comparison.csv",
                            mime="text/csv", key="shap_compare_download")
            except Exception as exc:
                st.error(f"Could not load SHAP comparison: {type(exc).__name__}: {exc}")
                st.caption("Check that both Drive files are shared with "
                           "the Streamlit service account.")


# The new-case workflow will reuse the original feature-only OpenSees generator.
# This first-stage panel verifies that the required source folders are reachable
# before enabling any predictions, avoiding fabricated or placeholder results.
NEW_PREDICTION_DRIVE_FOLDER_ID = "1E_nwCDgsw117Z7cI7w_BKRS8o_VXg4Gk"


def drive_list_children(folder_id: str, credentials_json: str) -> list:
    """List all children of a restricted Google Drive folder via service account."""
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    session = AuthorizedSession(credentials)
    files = []
    token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,size)",
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        response = session.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params, timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        files.extend(body.get("files", []))
        token = body.get("nextPageToken")
        if not token:
            break
    return files



@st.cache_data(show_spinner="Loading simulation matrix...", ttl=3600)
def load_new_prediction_matrix(folder_id: str, credentials_json: str) -> pd.DataFrame:
    """Download original input spreadsheet through the existing Drive service account."""
    items = drive_list_children(folder_id, credentials_json)
    matrix = next((item for item in items
                   if item["name"] == "SSI_Simulation_Matrix1(4).xlsx"), None)
    if matrix is None:
        raise FileNotFoundError("SSI_Simulation_Matrix1(4).xlsx not found in prediction folder")
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    response = AuthorizedSession(credentials).get(
        f"https://www.googleapis.com/drive/v3/files/{matrix['id']}",
        params={"alt": "media"}, timeout=120,
    )
    response.raise_for_status()
    df = pd.read_excel(io.BytesIO(response.content),
                       sheet_name="Simulation_Matrix", engine="openpyxl")
    df = df.dropna(subset=["Simulation_No"]).copy()
    df["Simulation_No"] = pd.to_numeric(df["Simulation_No"], errors="coerce")
    df = df.dropna(subset=["Simulation_No"])
    df["Simulation_No"] = df["Simulation_No"].astype(int)
    df = df.drop_duplicates(subset=["Simulation_No"])
    if df.empty:
        raise ValueError("No numbered simulations found in the original Excel matrix")
    return df


def input_from_matrix_row(row, column, label, key_prefix, *, positive=False,
                          whole=False):
    """Editable scalar engineering input, initialized from a selected Excel row."""
    value = float(row[column])
    if not np.isfinite(value):
        raise ValueError(f"Missing/invalid matrix input: {column}")
    key = f"{key_prefix}_{column}"
    if whole:
        # Existing simulation reader rounds spans/piles to integers.
        value = max(1, int(round(value)))
        return st.number_input(label, min_value=1, value=value, step=1, key=key)
    opts = {"min_value": 0.000001} if positive else {}
    if positive and value <= 0:
        raise ValueError(f"Expected positive input: {column}")
    return st.number_input(label, value=value, format="%.6f", key=key, **opts)



NEW_PREDICTION_FEATURES = {
    "Peak ground acceleration PGA (g)": "Target PGA (g)",
    "Column height H (m)": "Column height (m)",
    "Column/pile diameter D (m)": "Column/pile diameter (m)",
    "Span length (m)": "Span length (m)",
    "Number of spans (-)": "Number of spans",
    "Number of piles (-)": "Number of piles",
    "Pile spacing (m)": "Pile spacing (m)",
    "Pile embedment length Lp (m)": "Pile embedment (m)",
    "Longitudinal steel ratio pl (%)": "Longitudinal reinforcement (%)",
    "Transverse steel ratio pt (%)": "Transverse reinforcement (%)",
    "Concrete strength fc (MPa)": "Concrete strength (MPa)",
    "Steel yield strength fy (MPa)": "Steel yield strength (MPa)",
    "Angle of incidence (deg)": "Earthquake incidence angle (degrees)",
    "Ground motion time step (s)": "Ground motion time step (s)",
    "node_z_mean_m": "Mean structural node Z (m)",
    "node_z_min_m": "Minimum structural node Z (m)",
    "node_z_max_m": "Maximum structural node Z (m)",
}


def all_model_average(results, required_models=None):
    """Equal-weight average across ML model types, after averaging selected folds."""
    names = sorted(required_models if required_models is not None else results)
    if not names or any(name not in results for name in names):
        raise ValueError("All trained models must have successful predictions")
    means = np.asarray([results[name]["mean_m"] for name in names], dtype=float)
    if means.shape != (len(names), 2) or not np.isfinite(means).all():
        raise ValueError("Invalid trained-model predictions")
    # Per-fold averages allow the new-case scatter to use existing fold markers.
    fold_sets = [set(results[name]["folds"]) for name in names]
    if any(folds != fold_sets[0] for folds in fold_sets):
        raise ValueError("Selected models do not share the same folds")
    by_fold = {}
    for fold in sorted(fold_sets[0]):
        values = []
        for name in names:
            index = results[name]["folds"].index(fold)
            values.append(results[name]["predictions_m"][index])
        by_fold[fold] = np.asarray(values, dtype=float).mean(axis=0).tolist()
    return {"model_count": len(names), "mean_m": means.mean(axis=0).tolist(),
            "by_fold_m": by_fold}


def new_prediction_history_rows(history):
    """Create plot data only from real completed saved-checkpoint predictions."""
    rows = []
    for item in history:
        if not item.get("results"):
            continue
        for model_name, result in item["results"].items():
            for fold, values in zip(result["folds"], result["predictions_m"]):
                row = {
                    "Run": item["run"],
                    "Source simulation": item["source_simulation"],
                    "Model": model_name, "Fold": fold,
                    "Ux (mm)": 1000.0 * float(values[0]),
                    "Uy (mm)": 1000.0 * float(values[1]),
                }
                row.update(item.get("input_features", {}))
                row.update(item.get("graph_features", {}))
                rows.append(row)
        if set(item["results"]) == set(models):
            average = all_model_average(item["results"], sorted(models))
            for fold, values in average["by_fold_m"].items():
                row = {"Run": item["run"],
                       "Source simulation": item["source_simulation"],
                       "Model": "All-model average", "Fold": fold,
                       "Ux (mm)": 1000.0 * float(values[0]),
                       "Uy (mm)": 1000.0 * float(values[1])}
                row.update(item.get("input_features", {}))
                row.update(item.get("graph_features", {}))
                rows.append(row)
    return pd.DataFrame(rows)


def new_prediction_trend(data, feature, direction, model_filter, fold_filter,
                         runs_filter, ranges, x_axis_label=None):
    selected = data.loc[
        data["Model"].isin(model_filter)
        & data["Fold"].isin(fold_filter)
        & data["Run"].isin(runs_filter)
    ].copy()
    fig = go.Figure()
    fold_symbols = {1: "circle", 2: "square", 3: "diamond",
                    4: "triangle-up", 5: "triangle-down"}
    for (model_name, fold), part in selected.groupby(["Model", "Fold"], sort=True):
        fold_number = int(str(fold).replace("fold_", ""))
        fig.add_trace(go.Scatter(
            x=pd.to_numeric(part[feature], errors="coerce"),
            y=zero_small_displacements(
                pd.to_numeric(part[f"{direction} (mm)"], errors="coerce")),
            mode="markers", name=f"{model_name} / {fold}",
            marker={
                "symbol": fold_symbols.get(fold_number, "circle"),
                "size": 8 + 2 * fold_number,
                "color": "#111111" if model_name == "All-model average" else CASE_STUDY_COLORS.get(model_name, "#333333"),
                "line": {"color": "#222222", "width": 0.6},
            },
            customdata=part[["Run", "Source simulation"]].astype(str).to_numpy(),
            hovertemplate="Run %{customdata[0]}<br>Starting matrix row %{customdata[1]}"
                          "<br>Input %{x:.5g}<br>Prediction %{y:.5g} mm"
                          "<extra>%{fullData.name}</extra>",
        ))
    label = (x_axis_label or "").strip() or NEW_PREDICTION_FEATURES.get(feature, feature)
    return case_plot_style(fig, label, f"Predicted peak |{direction}| (mm)",
                           f"Predicted |{direction}| vs {label}", ranges)


with tab_new_prediction:
    st.subheader("New SSI response prediction")
    st.markdown("### Select and customize a new simulation")
    st.write(
        "Choose an original Excel matrix row as a starting template, then modify "
        "any engineering inputs. Edited values are for a new proposed case; "
        "the original Excel workbook is not changed."
    )
    if st.button("Load simulation matrix", key="newpred_load"):
        st.session_state["newpred_loaded"] = True
    if st.session_state.get("newpred_loaded", False):
        try:
            matrix_df = load_new_prediction_matrix(
                NEW_PREDICTION_DRIVE_FOLDER_ID,
                st.secrets["google_service_account_json"],
            )
            options = matrix_df["Simulation_No"].tolist()
            st.caption(f"{len(options):,} matrix rows available")
            if st.button("Choose a random matrix row", key="newpred_random"):
                st.session_state["newpred_case"] = int(np.random.default_rng().choice(options))
            if st.session_state.get("newpred_case") not in options:
                st.session_state["newpred_case"] = options[0]
            selected_id = st.selectbox(
                "Starting simulation row", options,
                key="newpred_case",
                format_func=lambda x: f"Simulation {x}",
            )
            row = matrix_df.loc[matrix_df["Simulation_No"].eq(selected_id)].iloc[0]
            prefix = f"newpred_edit_{selected_id}"
            edited = {"Simulation_No": int(selected_id)}
            left, right = st.columns(2)
            with left:
                st.markdown("**Structural geometry and configuration**")
                geometry = [
                    ("Column height H (m)", "Column height H (m)", True, False),
                    ("Column/pile diameter D (m)", "Column/pile diameter D (m)", True, False),
                    ("Number of spans", "Number of spans (-)", True, True),
                    ("Span length (m)", "Span length (m)", True, False),
                    ("Number of piles", "Number of piles (-)", True, True),
                    ("Pile spacing (m)", "Pile spacing (m)", True, False),
                    ("Pile embedment length Lp (m)", "Pile embedment length Lp (m)", True, False),
                ]
                for label, column, positive, whole in geometry:
                    edited[column] = input_from_matrix_row(
                        row, column, label, prefix, positive=positive, whole=whole)
                edited["Number of bents (-)"] = int(edited["Number of spans (-)"]) + 1
                st.caption(f"Calculated number of bents: {edited['Number of bents (-)']}")
                st.markdown("**Concrete and steel**")
                materials = [
                    ("Longitudinal reinforcement ratio (%)", "Longitudinal steel ratio pl (%)"),
                    ("Transverse reinforcement ratio (%)", "Transverse steel ratio pt (%)"),
                    ("Concrete strength fc (MPa)", "Concrete strength fc (MPa)"),
                    ("Steel yield strength fy (MPa)", "Steel yield strength fy (MPa)"),
                ]
                for label, column in materials:
                    edited[column] = input_from_matrix_row(
                        row, column, label, prefix, positive=True)
            with right:
                st.markdown("**Soil and earthquake selection**")
                edited["Soil Profile (based on SPT-N)"] = st.text_input(
                    "Soil profile (existing profile filename/stem)",
                    value=str(row["Soil Profile (based on SPT-N)"]),
                    key=f"{prefix}_soil")
                edited["EQ"] = st.text_input(
                    "Earthquake record (existing record identifier)",
                    value=str(row["EQ"]),
                    key=f"{prefix}_eq")
                st.caption(
                    "Soil profile and earthquake identifiers must match the files "
                    "in the connected Google Drive folders."
                )
                st.markdown("**Earthquake loading properties**")
                edited["Peak ground acceleration PGA (g)"] = input_from_matrix_row(
                    row, "Peak ground acceleration PGA (g)",
                    "Target PGA (g)", prefix, positive=True)
                edited["Ground motion time step (s)"] = input_from_matrix_row(
                    row, "Ground motion time step (s)",
                    "Ground-motion time step (s)", prefix, positive=True)
                edited["Angle of incidence (deg)"] = input_from_matrix_row(
                    row, "Angle of incidence (deg)",
                    "Incidence angle (degrees)", prefix)

            st.markdown("### Generate features and predict displacement")
            active_models = sorted(models)
            st.caption(f"Using all {len(active_models)} trained ML models for the overall average.")
            active_folds = st.selectbox(
                "Folds to predict",
                ["5-fold ensemble", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
                key="new_case_folds_select")
            st.caption(
                "The original OpenSees code constructs the structure and its "
                "input graph only. Dynamic analysis and ML training are not run.")
            if st.button("Generate Features & Predict Displacement",
                         type="primary", key="new_case_predict_button"):
                if not active_models:
                    st.error("Select at least one trained ML model.")
                else:
                    try:
                        from new_case_engine import build_new_case
                        with st.spinner("Building OpenSees input graph and running trained checkpoints..."):
                            input_file, new_case_id, temp_owner, graph_stats = build_new_case(
                                edited, int(selected_id),
                                NEW_PREDICTION_DRIVE_FOLDER_ID,
                                st.secrets["google_service_account_json"],
                                Path(__file__).resolve().parent,
                            )
                            try:
                                case_results = {}
                                for name in active_models:
                                    case_results[name] = predict_folds(
                                        input_file, new_case_id, root, name,
                                        fold=None if active_folds == "5-fold ensemble"
                                        else active_folds)
                            finally:
                                temp_owner.cleanup()
                        stored = st.session_state.setdefault("new_case_history", [])
                        run_number = 1 + max((int(x.get("run", 0)) for x in stored),
                                             default=0)
                        stored.append({
                            "run": run_number,
                            "source_simulation": int(selected_id),
                            "input_features": dict(edited),
                            "graph_features": graph_stats,
                            "results": case_results,
                        })
                        st.success(
                            f"Prediction run {run_number} completed and automatically "
                            "added to the cumulative comparison plots below.")
                        preview = []
                        for model_name, result in case_results.items():
                            preview.append({
                                "Model": model_name,
                                "Ux (mm)": 1000.0 * result["mean_m"][0],
                                "Uy (mm)": 1000.0 * result["mean_m"][1],
                                "Fold SD Ux (mm)": 1000.0 * result["fold_std_m"][0],
                                "Fold SD Uy (mm)": 1000.0 * result["fold_std_m"][1],
                            })
                        combined = all_model_average(case_results, active_models)
                        avg_ux, avg_uy = combined["mean_m"]
                        st.markdown("#### Average prediction across all ML models")
                        kx, ky = st.columns(2)
                        kx.metric("All-model average |Ux| (mm)", f"{1000 * avg_ux:.4f}")
                        ky.metric("All-model average |Uy| (mm)", f"{1000 * avg_uy:.4f}")
                        preview.append({
                            "Model": "All-model average",
                            "Ux (mm)": 1000 * avg_ux, "Uy (mm)": 1000 * avg_uy,
                            "Fold SD Ux (mm)": np.nan, "Fold SD Uy (mm)": np.nan,
                        })
                        st.dataframe(pd.DataFrame(preview), hide_index=True,
                                     use_container_width=True)
                        st.caption("Each of the seven trained model types receives equal weight. "
                                   "For a five-fold ensemble, each model is averaged over its "
                                   "five folds first. This is not a calibrated confidence interval.")
                    except Exception as exc:
                        st.error(f"New-case prediction failed: {type(exc).__name__}: {exc}")

            with st.expander("Parametric study — sweep one input parameter", expanded=False):
                st.write(
                    "Vary one parameter across a specified range while keeping "
                    "all other currently edited inputs fixed. Each successful "
                    "case is automatically added to cumulative displacement plots.")
                sweep_options = {
                    key: label for key, label in NEW_PREDICTION_FEATURES.items()
                    if key in edited and isinstance(edited[key], (int, float))
                    and not isinstance(edited[key], bool)
                }
                sweep_key = st.selectbox(
                    "Parameter to vary", list(sweep_options),
                    format_func=lambda name: sweep_options[name],
                    key="new_case_sweep_feature")
                current_value = float(edited[sweep_key])
                sweep_is_integer = sweep_key in (
                    "Number of spans (-)", "Number of piles (-)")
                sweep_defaults = st.session_state.get("new_case_sweep_last_feature")
                if sweep_defaults != sweep_key:
                    st.session_state["new_case_sweep_min"] = current_value
                    st.session_state["new_case_sweep_max"] = current_value + (
                        2.0 if sweep_is_integer else max(0.2 * abs(current_value), 0.1))
                    st.session_state["new_case_sweep_step"] = (
                        1.0 if sweep_is_integer else max(0.1 * abs(current_value), 0.05))
                    st.session_state["new_case_sweep_last_feature"] = sweep_key
                sweep_left, sweep_mid, sweep_right = st.columns(3)
                with sweep_left:
                    sweep_min = st.number_input(
                        "Minimum value", format="%.6f",
                        key="new_case_sweep_min")
                with sweep_mid:
                    sweep_max = st.number_input(
                        "Maximum value", format="%.6f",
                        key="new_case_sweep_max")
                with sweep_right:
                    sweep_step = st.number_input(
                        "Interval (step)", min_value=0.000001, format="%.6f",
                        key="new_case_sweep_step")
                sweep_values = []
                sweep_problem = None
                if sweep_max < sweep_min:
                    sweep_problem = "Maximum value must be greater than or equal to minimum."
                elif sweep_step <= 0:
                    sweep_problem = "Step must be positive."
                else:
                    count = int(np.floor(
                        (sweep_max - sweep_min) / sweep_step + 1e-9)) + 1
                    if count > 12:
                        sweep_problem = (
                            f"Your range would create {count} cases. "
                            "Use a larger interval or a smaller range (maximum 12 per batch).")
                    elif count < 2:
                        sweep_problem = "Choose a range and interval that produce at least two cases."
                    else:
                        sweep_values = [
                            round(sweep_min + i * sweep_step, 8)
                            for i in range(count)
                        ]
                        if sweep_is_integer and any(
                                abs(value - round(value)) > 1e-7
                                for value in sweep_values):
                            sweep_problem = (
                                "This parameter requires whole-number minimum and interval.")
                        elif any(value <= 0 for value in sweep_values) and sweep_key not in (
                                "Angle of incidence (deg)",):
                            sweep_problem = "The selected parameter requires positive values."
                        elif (sweep_key == "Angle of incidence (deg)"
                              and any(not 0 <= value <= 360 for value in sweep_values)):
                            sweep_problem = "Incidence angle must remain between 0 and 360 degrees."
                if sweep_problem:
                    st.warning(sweep_problem)
                else:
                    st.caption(
                        f"{len(sweep_values)} cases: "
                        + ", ".join(f"{value:g}" for value in sweep_values)
                        + f" {sweep_options[sweep_key]}. All other input values remain unchanged.")
                    if sweep_key == "Pile embedment length Lp (m)" and any(
                            value <= float(edited["Column height H (m)"])
                            for value in sweep_values):
                        sweep_problem = (
                            "Pile embedment length must exceed the current column height.")
                        st.warning(sweep_problem)
                if st.button(
                        f"Generate features and predict {len(sweep_values)} sweep cases",
                        key="new_case_run_parametric",
                        disabled=bool(sweep_problem) or not sweep_values,
                        type="primary"):
                    from new_case_engine import build_new_case
                    progress = st.progress(0.0)
                    progress_detail = st.empty()
                    sweep_summary = []
                    for sweep_idx, sweep_value in enumerate(sweep_values):
                        progress_detail.info(
                            f"Case {sweep_idx + 1}/{len(sweep_values)} — "
                            f"{sweep_options[sweep_key]} = {sweep_value:g}")
                        sweep_inputs = dict(edited)
                        sweep_inputs[sweep_key] = (
                            int(round(sweep_value)) if sweep_is_integer else float(sweep_value))
                        if sweep_key == "Number of spans (-)":
                            sweep_inputs["Number of bents (-)"] = (
                                int(sweep_inputs[sweep_key]) + 1)
                        try:
                            input_file, new_case_id, temp_owner, graph_stats = build_new_case(
                                sweep_inputs, int(selected_id),
                                NEW_PREDICTION_DRIVE_FOLDER_ID,
                                st.secrets["google_service_account_json"],
                                Path(__file__).resolve().parent)
                            try:
                                case_results = {}
                                for name in active_models:
                                    case_results[name] = predict_folds(
                                        input_file, new_case_id, root, name,
                                        fold=None if active_folds == "5-fold ensemble"
                                        else active_folds)
                            finally:
                                temp_owner.cleanup()
                            stored = st.session_state.setdefault("new_case_history", [])
                            run_number = 1 + max(
                                (int(x.get("run", 0)) for x in stored), default=0)
                            stored.append({
                                "run": run_number,
                                "source_simulation": int(selected_id),
                                "input_features": sweep_inputs,
                                "graph_features": graph_stats,
                                "results": case_results,
                            })
                            averaged = all_model_average(case_results, active_models)["mean_m"]
                            sweep_summary.append({
                                "Parameter": sweep_options[sweep_key],
                                "Input value": sweep_value,
                                "Run": run_number,
                                "Average Ux (mm)": 1000.0 * averaged[0],
                                "Average Uy (mm)": 1000.0 * averaged[1],
                                "Status": "Predicted",
                            })
                        except Exception as exc:
                            sweep_summary.append({
                                "Parameter": sweep_options[sweep_key],
                                "Input value": sweep_value,
                                "Run": None,
                                "Average Ux (mm)": None,
                                "Average Uy (mm)": None,
                                "Status": f"Failed: {type(exc).__name__}: {exc}",
                            })
                        progress.progress((sweep_idx + 1) / len(sweep_values))
                    successes = sum(row["Status"] == "Predicted" for row in sweep_summary)
                    progress_detail.success(
                        f"Parametric study finished: {successes}/{len(sweep_values)} "
                        "cases predicted and added to the cumulative plot.")
                    st.dataframe(pd.DataFrame(sweep_summary),
                                 hide_index=True, use_container_width=True)
                    if successes < len(sweep_values):
                        st.warning(
                            "Some cases failed validation or feature generation; "
                            "successful cases remain in the history below.")
                    st.session_state["new_case_trend_input"] = sweep_key

            st.markdown("**Customized simulation inputs**")
            st.dataframe(
                pd.DataFrame(
                    [{"Parameter": name, "Value": value}
                     for name, value in edited.items() if name != "Simulation_No"]
                ),
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                "Download customized input parameters (JSON)",
                data=json.dumps({
                    "source_simulation": int(selected_id),
                    "edited_matrix_parameters": edited,
                    "note": "New case; not an OpenSees dynamic result or an ML prediction",
                }, indent=2),
                file_name=f"new_ssi_case_from_simulation_{selected_id}.json",
                mime="application/json", key="newpred_download_inputs",
            )
            st.info(
                "These edited values are ready to be passed to the original "
                "simulation-matrix reader and feature-only OpenSees builder. "
                "The actual feature-generation and prediction execution is "
                "not enabled yet; no prediction is being claimed."
            )
        except Exception as exc:
            st.error(f"Could not prepare editable matrix row: {type(exc).__name__}: {exc}")


    st.markdown("### Cumulative predictions and input-response trends")
    history = st.session_state.get("new_case_history", [])
    st.caption(
        f"{len(history)} successful prediction runs in this session. "
        "Every new successful prediction is automatically added; previous "
        "points are not replaced.")
    history_table = new_prediction_history_rows(history)
    if not history_table.empty:
        st.dataframe(history_table, hide_index=True, use_container_width=True)
        feature_choices = [
            key for key in NEW_PREDICTION_FEATURES
            if key in history_table.columns
            and pd.to_numeric(history_table[key], errors="coerce").notna().any()
        ]
        fcol, mcol, foldcol = st.columns(3)
        with fcol:
            x_key = st.selectbox(
                "Input feature for X axis", feature_choices,
                format_func=lambda key: NEW_PREDICTION_FEATURES[key],
                key="new_case_trend_input")
        with mcol:
            model_options = sorted(history_table["Model"].unique())
            chosen_models = st.multiselect(
                "Models to plot", model_options, default=model_options,
                key="new_case_trend_models")
        with foldcol:
            fold_options = sorted(history_table["Fold"].unique())
            chosen_folds = st.multiselect(
                "Folds to plot", fold_options, default=fold_options,
                key="new_case_trend_folds")
        custom_x_label = st.text_input(
            "Custom X-axis label (optional)",
            value="",
            placeholder=NEW_PREDICTION_FEATURES.get(x_key, x_key),
            help="Leave blank to use the selected engineering feature name. "
                 "Changes the displayed X-axis label, not the plotted data.",
            key="new_case_custom_x_axis_label")
        run_options = sorted(history_table["Run"].unique().tolist())
        chosen_runs = st.multiselect(
            "Simulation runs to plot", run_options, default=run_options,
            key="new_case_trend_runs")
        with st.expander("Custom X/Y axis limits", expanded=False):
            manual = st.checkbox("Set manual axis ranges", key="new_case_axes_manual")
            limits = None
            if manual:
                left_axis, right_axis = st.columns(2)
                with left_axis:
                    xmin = st.number_input("X minimum", value=0.0, key="new_case_xmin")
                    xmax = st.number_input("X maximum", value=10.0, key="new_case_xmax")
                with right_axis:
                    ymin = st.number_input("Y minimum (mm)", value=0.0,
                                            key="new_case_ymin")
                    ymax = st.number_input("Y maximum (mm)", value=10.0,
                                            key="new_case_ymax")
                if xmax > xmin and ymax > ymin:
                    limits = ((xmin, xmax), (ymin, ymax))
                else:
                    st.warning("Maximum limits must exceed minimum limits.")
        if (chosen_models and chosen_folds and chosen_runs and
                (not manual or limits is not None)):
            ux_tab_new, uy_tab_new = st.tabs([
                "Predicted Ux vs input", "Predicted Uy vs input"])
            for direction, panel in (("Ux", ux_tab_new), ("Uy", uy_tab_new)):
                with panel:
                    st.plotly_chart(
                        new_prediction_trend(
                            history_table, x_key, direction,
                            chosen_models, chosen_folds, chosen_runs, limits,
                            x_axis_label=custom_x_label),
                        use_container_width=True,
                        key=f"new_case_trend_plot_{direction}")
        st.download_button(
            "Download all predicted cases (CSV)",
            history_table.to_csv(index=False).encode("utf-8"),
            file_name="new_ssi_predictions.csv", mime="text/csv",
            key="new_case_history_csv")
        st.download_button(
            "Save cumulative history (JSON)",
            json.dumps(history, indent=2),
            file_name="new_ssi_prediction_history.json",
            mime="application/json", key="new_case_history_json")
        if st.button("Clear prediction history", key="new_case_clear_history"):
            st.session_state["new_case_history"] = []
            st.rerun()
    uploaded_history = st.file_uploader(
        "Restore earlier prediction history (JSON backup)", type=["json"],
        key="new_case_history_uploader")
    if uploaded_history is not None and st.button(
            "Restore history", key="new_case_restore_history"):
        try:
            candidate = json.loads(uploaded_history.getvalue().decode("utf-8"))
            if (not isinstance(candidate, list) or
                    any(not isinstance(record, dict)
                        or not {"run", "source_simulation", "input_features",
                                "results"}.issubset(record)
                        for record in candidate)):
                raise ValueError("File does not contain saved new-case prediction history")
            st.session_state["new_case_history"] = candidate
            st.rerun()
        except Exception as exc:
            st.error(f"Could not restore history: {exc}")
    st.caption(
        "Prediction history remains available during the active Streamlit session. "
        "Download JSON and restore it after closing or refreshing the app. "
        "Points represent checkpoint inference, not OpenSees dynamic results.")

    st.markdown("#### ML model input features")
    st.caption(
        "Counts are based on the deployed training/inference feature definitions. "
        "Feature channels are not the same as user-entered parameters; node and edge "
        "values are generated from the OpenSees model."
    )
    f1, f2, f3 = st.columns(3)
    f1.metric("Engineering/global parameters", "19")
    f2.metric("Graph feature channels", "15")
    f3.metric("Earthquake feature channels", "67")
    st.dataframe(pd.DataFrame([
        {"Input category": "Structural geometry and layout",
         "Number": 7, "Includes": "Lengths, diameters, spans and pile geometry"},
        {"Input category": "Concrete and steel properties",
         "Number": 3, "Includes": "Reinforcement and material strengths"},
        {"Input category": "Soil properties and layering",
         "Number": 5, "Includes": "Soil-layer count and weighted soil properties"},
        {"Input category": "Ground-motion setup",
         "Number": 4, "Includes": "Target PGA, incidence direction and time step"},
        {"Input category": "Structural graph: nodes",
         "Number": 6, "Includes": "Coordinates and node/topology indicators"},
        {"Input category": "Structural graph: edges",
         "Number": 9, "Includes": "Element length, connectivity and element type"},
        {"Input category": "Earthquake: whole-record descriptors",
         "Number": 47, "Includes": "PGA, RMS, Arias, FFT, Prony frequency/damping and duration"},
        {"Input category": "Earthquake: window descriptor channels",
         "Number": 20, "Includes": "Amplitude, energy and frequency measures per time window"},
    ]), hide_index=True, use_container_width=True)
    st.info(
        "**Neural ML models:** 101 feature channels overall, with earthquake "
        "window channels evaluated across 24 ordered windows (480 values). "
        "**GPR:** 606 flattened input values, including node/edge statistics "
        "and all earthquake windows. These are two representations of the "
        "same underlying engineering inputs, not 606 manual inputs."
    )
    st.write(
        "Prepare a new simulation-matrix row, generate the OpenSees model in "
        "feature-only mode, and use saved checkpoints to predict peak |Ux| and |Uy|. "
        "No dynamic analysis or retraining is required."
    )
    st.caption(
        "Setup stage: verify the required Google Drive source files. "
        "New-case inference will be enabled after original feature generation "
        "and HDF5 compatibility are validated."
    )
    if st.button("Check prediction input folders", type="primary",
                 key="new_prediction_check_folders"):
        try:
            credentials_json = st.secrets["google_service_account_json"]
            children = drive_list_children(
                NEW_PREDICTION_DRIVE_FOLDER_ID, credentials_json
            )
            found = {item["name"]: item for item in children}
            required = ["SSI_Simulation_Matrix1(4).xlsx", "EQs", "soil_profiles"]
            rows = []
            for name in required:
                item = found.get(name)
                expected_folder = name in ("EQs", "soil_profiles")
                valid_type = item is not None and (
                    (item.get("mimeType") == "application/vnd.google-apps.folder")
                    if expected_folder else
                    (item.get("mimeType") != "application/vnd.google-apps.folder")
                )
                rows.append({"Required input": name,
                             "Status": "Found" if valid_type else "Missing or wrong type"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            for name in ("EQs", "soil_profiles"):
                folder = found.get(name)
                if folder and folder.get("mimeType") == "application/vnd.google-apps.folder":
                    contents = drive_list_children(folder["id"], credentials_json)
                    st.write(f"{name}: {len(contents)} immediate entries found")
            if all(row["Status"] == "Found" for row in rows):
                st.success("Input matrix and both source folders are accessible.")
            else:
                st.warning("Verify filenames, folder structure, and inherited Viewer access.")
        except Exception as exc:
            st.error(f"Google Drive access check failed: {type(exc).__name__}: {exc}")


with tab_about:
    st.header("About the ML surrogate and its inputs")
    st.write(
        "This application predicts the maximum absolute horizontal displacement "
        "at the first response-recorder node in the X and Y directions (Ux, Uy). "
        "The seven trained surrogate models use engineering information from "
        "a bridge–soil OpenSees model and derived earthquake descriptors."
    )
    st.caption(
        "Counts below refer to input feature channels, not the number of independent "
        "parameters a user must type. Some values are automatically computed."
    )

    st.subheader("Engineering and structural inputs — 34 channels")
    with st.expander("Structural geometry and layout — 7", expanded=True):
        st.write(
            "Describes the physical size and arrangement of the bridge and pile foundation: "
            "column height and diameter; number and length of spans; number of piles, "
            "pile spacing, and pile embedment length. These dimensions determine "
            "the structural configuration generated by OpenSees."
        )
    with st.expander("Concrete and steel properties — 3"):
        st.write(
            "Describes column reinforcement percentage, concrete compressive strength "
            "and steel yield strength. These values influence structural stiffness "
            "and resistance. Other OpenSees material definitions may be used to "
            "build the model, but are not separate global ML feature channels."
        )
    with st.expander("Soil properties and layering — 5"):
        st.write(
            "Describes the number of soil layers and thickness-weighted summaries "
            "of soil unit weight (gamma), friction angle (phi), undrained shear "
            "strength (su), and shear modulus (G). Layer-by-layer soil properties "
            "are also involved in generating the OpenSees structure and its topology; "
            "the ML global vector uses these five summaries."
        )
    with st.expander("Ground-motion setup — 4"):
        st.write(
            "Specifies the target peak ground acceleration (PGA), the sine and cosine "
            "of the earthquake incidence angle (its two directional components), "
            "and the acceleration sampling time step. The selected earthquake "
            "record is scaled and projected along global X and Y before "
            "earthquake descriptors are extracted."
        )
    with st.expander("OpenSees nodal/graph channels — 6"):
        st.write(
            "The generated structural graph stores three-dimensional nodal "
            "coordinates (x, y, z), graph degree (how many directed element "
            "connections a node has), a pile/column-node indicator, and a "
            "pile-bottom-node indicator. Each of the six values is a channel "
            "repeated for the relevant nodes, not six values for the entire bridge."
        )
    with st.expander("OpenSees element/edge channels — 9"):
        st.write(
            "For each directed structural connection, the graph stores element "
            "length, the number of nodes connected by its parent element, and "
            "seven one-hot element-type indicators: dispBeamColumn, "
            "elasticBeamColumn, zeroLength, twoNodeLink, forceBeamColumn, "
            "SSPbrick, or other. Thus the nine channels repeat across edges."
        )

    st.subheader("Earthquake descriptors — 67 channels")
    with st.expander("Whole-record earthquake descriptors — 47", expanded=True):
        st.write(
            "These summarize the complete two-component ground-acceleration "
            "record, rather than passing every acceleration sample directly "
            "to the trained model. For each horizontal direction (X and Y), "
            "22 descriptors are calculated:"
        )
        st.markdown(
            "- **Motion amplitude and cumulative intensity (4 per direction):** "
            "PGA, root-mean-square acceleration (RMS), Arias intensity, "
            "and cumulative absolute velocity (CAV).\n"
            "- **Strong-motion duration (2):** D5–95 and D5–75 energy-duration measures.\n"
            "- **Fourier frequency content (7):** dominant frequency, spectral "
            "centroid, bandwidth, and relative spectral energy in four "
            "frequency bands (0–1, 1–3, 3–6, and above 6 Hz).\n"
            "- **Prony-based descriptors (9):** for up to three identified "
            "components, frequency, damping ratio, and relative amplitude."
        )
        st.write(
            "The remaining three descriptors are correlation between the "
            "X and Y accelerations, the sampling time step, and record duration. "
            "**Count: 22 × 2 + 3 = 47.**"
        )
        st.caption(
            "Prony damping ratios describe damped components estimated from the "
            "ground-motion signal; they are not independently specified "
            "structural damping ratios."
        )
    with st.expander("Earthquake time-window descriptors — 20", expanded=True):
        st.write(
            "The acceleration history is divided into **24 ordered time windows**. "
            "In every window, 10 quantities are computed separately for X and Y:"
        )
        st.markdown(
            "- **Amplitude (3 per direction):** RMS acceleration, peak absolute "
            "acceleration, and mean absolute acceleration.\n"
            "- **Energy (1):** Arias energy for that window.\n"
            "- **Frequency content (6):** dominant frequency, FFT spectral "
            "centroid, and spectral-energy fractions in four frequency bands "
            "(0–1, 1–3, 3–6, and above 6 Hz)."
        )
        st.write(
            "**Count: 10 per direction × 2 directions = 20 channels per window.** "
            "With 24 windows, the model processes **24 × 20 = 480 ordered "
            "earthquake-window values**. The time order helps represent "
            "how ground-motion intensity and frequency content evolve."
        )

    st.subheader("How feature counts differ between model families")
    st.table(pd.DataFrame([
        {"Representation": "Neural surrogate models",
         "Inputs": "19 global + 6 node + 9 edge + 47 record + 20 window channels",
         "Count": "101 feature channels"},
        {"Representation": "Earthquake window sequence",
         "Inputs": "24 windows × 20 descriptors",
         "Count": "480 ordered values"},
        {"Representation": "Gaussian Process Regression (GPR)",
         "Inputs": "66 global/record + 24 node statistics + 36 edge statistics + 480 window values",
         "Count": "606 scalar values"},
    ]))
    st.caption(
        "GPR turns each node/edge channel into four summary statistics "
        "(mean, standard deviation, minimum and maximum). The neural models "
        "instead retain graph and time-window structures. The input counts "
        "describe different representations of the same bridge and earthquake case."
    )
    st.info(
        "New-case prediction requires constructing the same OpenSees input graph "
        "and earthquake features used during training. Predicted Ux/Uy are "
        "surrogate estimates, not independently measured or simulated responses."
    )
