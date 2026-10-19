"""SSI saved-checkpoint inference with Google Drive HDF5 and upload alternatives."""
import json
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

tab_predict, tab_compare, tab_overfit, tab_case_study = st.tabs([
    "Predict a simulation",
    "Actual vs. predicted — 780 cases",
    "Overfitting analysis — train / validation / test",
    "Case study & SHAP analysis",
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


def plot_comparison(data: pd.DataFrame, direction: str, title: str, axis_ranges=None, legend_rmse=False):
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
                    "color": colors.get(model, "#555555")},
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
                fold_detail = st.checkbox(
                    "Show each selected fold separately in scatter plots",
                    value=False, key="overfit_fold_detail")
                if not chosen_models or not chosen_folds or not selected_splits:
                    st.info("Select at least one model, fold, and dataset.")
                else:
                    selected = of[
                        of["model"].isin(chosen_models)
                        & of["fold"].isin(chosen_folds)
                        & of["split"].isin(selected_splits)
                    ].copy()
                    scatter_data = overfit_plot_data(selected, fold_detail)
                    ux_view, uy_view = st.tabs(["Training / validation / test — Ux",
                                                "Training / validation / test — Uy"])
                    for direction, area in [("Ux", ux_view), ("Uy", uy_view)]:
                        with area:
                            st.markdown(f"### Peak |{direction}| performance")
                            st.caption("Each scatter series represents one model/dataset"
                                       + ("/fold." if fold_detail else "; points across selected folds are pooled."))
                            st.caption("Legend: RMSE (mm), calculated for each displayed series. "
                                       "R², MAPE, and Error SD are available in the tables below.")
                            custom_ranges = overfit_axis_controls(direction)
                            plot, _ = plot_comparison(
                                scatter_data, direction,
                                f"Peak |{direction}| — training vs. validation vs. test",
                                axis_ranges=custom_ranges, legend_rmse=True)
                            st.plotly_chart(plot, use_container_width=True,
                                            key=f"overfit_plot_{direction}")
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
                                st.caption("Higher relative scores are better. Each model keeps its own marker "
                                           "and color; training: dotted, validation: dashed, test: solid. "
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


def case_plot_style(fig, x_title, y_title, title):
    fig.update_layout(
        template="plotly_white", height=510, paper_bgcolor="white",
        plot_bgcolor="white",
        title={"text": title, "font": {"color": "#000000", "size": 18}},
        font={"family": CASE_STUDY_FONT, "color": "#000000", "size": 13},
        xaxis_title=x_title, yaxis_title=y_title,
        legend={"x": 1.02, "y": 1, "xanchor": "left",
                "font": {"color": "#000000", "size": 11}},
        margin={"l": 85, "r": 190, "t": 75, "b": 85})
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(showgrid=True, gridcolor="#D0D0D0", griddash="dash",
             tickfont={"color": "#000000"},
             title_font={"color": "#000000"})
    return fig


def case_study_scatter(frame, feature_name, simulation, direction, color,
                       feature_stat):
    """Actual/predicted displacement vs the chosen raw input summary."""
    f = frame[(frame["feature"] == feature_name) &
              (frame["output"] == direction)].copy()
    fig = go.Figure()
    if f.empty:
        return fig
    x = f[feature_stat].to_numpy(dtype=float)
    for label, col, marker in (
        ("Actual", f"actual_{direction}_mm", "circle-open"),
        ("Predicted", f"predicted_{direction}_mm", "diamond-open"),
    ):
        y = zero_small_displacements(f[col].to_numpy(dtype=float))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=label,
            text=f["simulation_id"], customdata=f[["split"]].to_numpy(),
            marker={"symbol": marker, "size": 10, "color": color,
                    "line": {"width": 1.5, "color": color}},
            hovertemplate="Case %{text}<br>Split: %{customdata[0]}" +
                          "<br>Feature summary: %{x:.5g}<br>Displacement: %{y:.5g} mm" +
                          "<extra>%{fullData.name}</extra>"))
        marked = f["simulation_id"].eq(simulation)
        if marked.any():
            point = f.loc[marked].iloc[0]
            px = float(point[feature_stat])
            py = float(zero_small_displacements([point[col]])[0])
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers",
                name=f"Selected case — {label}",
                marker={"symbol": "star", "size": 15,
                        "color": "#000000", "line": {"color": color, "width": 2}},
                text=[simulation], hovertemplate="%{text}<br>%{y:.5g} mm<extra>%{fullData.name}</extra>"))
    return case_plot_style(
        fig, f"{feature_name} ({feature_stat.replace('input_feature_', '')})",
        f"Peak |{direction}| displacement (mm)",
        f"Input feature vs actual/predicted |{direction}|")


def case_shap_scatter(frame, feature_name, simulation, direction, color, feature_stat):
    """Sampled grouped Shapley contribution (metres converted to millimetres)."""
    f = frame[(frame["feature"] == feature_name) &
              (frame["output"] == direction)].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f[feature_stat], y=f["shapley_value"] * 1000,
        mode="markers", name="SHAP contribution",
        text=f["simulation_id"],
        marker={"symbol": "circle-open", "size": 10, "color": color,
                "line": {"width": 1.5, "color": color}},
        hovertemplate="Case %{text}<br>Feature: %{x:.5g}" +
                      "<br>Contribution: %{y:.5g} mm<extra></extra>"))
    marked = f["simulation_id"].eq(simulation)
    if marked.any():
        point = f.loc[marked].iloc[0]
        fig.add_trace(go.Scatter(
            x=[point[feature_stat]], y=[point["shapley_value"] * 1000],
            mode="markers", name="Selected case",
            marker={"symbol": "star", "size": 15, "color": "#000000"},
            hovertemplate="Selected case<br>SHAP: %{y:.5g} mm<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color="#505050")
    return case_plot_style(
        fig, f"{feature_name} ({feature_stat.replace('input_feature_', '')})",
        f"Sampled Shapley contribution to |{direction}| (mm)",
        f"Input feature vs SHAP contribution — |{direction}|")


with tab_case_study:
    st.subheader("Case study — input features, SHAP contributions and peak displacements")
    st.caption("Separate from the other dashboard tabs. SHAP sample data explain selected "
               "held-out simulations, not all 780 simulations. SHAP values are approximate "
               "and shown in mm (original contributions are in metres).")
    st.caption("For grouped node, edge, and earthquake-window channels, the input on the "
               "horizontal axis is a raw channel summary across nodes/edges/time windows, "
               "not an individual scalar value.")
    try:
        shap_creds = st.secrets["google_service_account_json"]
        shap_id = st.secrets["google_drive"].get(
            "shap_samples_file_id", SHAP_SAMPLES_DRIVE_ID)
        prediction_id = st.secrets["google_drive"].get(
            "overfitting_file_id", OVERFITTING_DRIVE_FILE_ID)
    except (KeyError, FileNotFoundError):
        st.error("Missing Google service-account credentials in Streamlit Secrets.")
    else:
        if (st.button("Load case-study data", key="load_shap_case")
                or st.session_state.get("shap_case_loaded", False)):
            try:
                shap_data = load_shap_samples(shap_id, shap_creds)
                prediction_data = load_comparison_csv(prediction_id, shap_creds)
                if "split" not in prediction_data.columns:
                    raise ValueError("Split labels are required in the prediction CSV.")
                join = ["simulation_id", "model", "fold"]
                sample_keys = shap_data[join].drop_duplicates()
                if prediction_data.duplicated(join).any():
                    raise ValueError("Prediction CSV has duplicate simulation/model/fold keys.")
                comparison = prediction_data.merge(sample_keys, on=join, how="inner")
                if comparison.empty:
                    raise ValueError("No matching simulation/model/fold combinations between SHAP and predictions.")
                shap_data = shap_data.merge(
                    comparison[join], on=join, how="inner")
                st.session_state["shap_case_loaded"] = True
                st.caption(f"Matched {comparison.simulation_id.nunique()} explained simulations "
                           f"across {len(comparison)} model/fold predictions.")
                c1, c2, c3 = st.columns(3)
                model_selected = c1.selectbox(
                    "ML model", sorted(shap_data["model"].unique()), key="shap_case_model")
                available_folds = sorted(shap_data.loc[
                    shap_data["model"].eq(model_selected), "fold"].unique())
                fold_selected = c2.selectbox(
                    "Fold", available_folds, key="shap_case_fold")
                sample_subset = shap_data.loc[
                    shap_data["model"].eq(model_selected) &
                    shap_data["fold"].eq(fold_selected)].copy()
                feature_selected = c3.selectbox(
                    "Input feature", sorted(sample_subset["feature"].unique()),
                    key="shap_case_feature")
                cases_for_feature = sorted(sample_subset.loc[
                    sample_subset["feature"].eq(feature_selected), "simulation_id"].unique())
                selected_simulation = st.selectbox(
                    "Highlighted simulation", cases_for_feature, key="shap_case_simulation")
                summary_kind = st.selectbox(
                    "Input feature statistic (X axis)",
                    ["mean", "min", "max", "std"],
                    key="shap_case_stat")
                stat_col = "input_feature_" + summary_kind
                p = comparison.loc[
                    comparison["model"].eq(model_selected) &
                    comparison["fold"].eq(fold_selected)].copy()
                sub = sample_subset.merge(
                    p[join + ["split", "actual_Ux_mm", "predicted_Ux_mm",
                              "actual_Uy_mm", "predicted_Uy_mm"]],
                    on=join, how="inner", validate="many_to_one")
                target = p.loc[p["simulation_id"].eq(selected_simulation)]
                if target.empty:
                    st.warning("Selected SHAP simulation has no matching displacement record.")
                else:
                    row = target.iloc[0]
                    st.markdown(f"**{selected_simulation}** | {model_selected} | "
                                f"{fold_selected} | {row['split']}")
                    a1, a2, a3, a4 = st.columns(4)
                    for area, label, column in (
                        (a1, "Actual Ux (mm)", "actual_Ux_mm"),
                        (a2, "Predicted Ux (mm)", "predicted_Ux_mm"),
                        (a3, "Actual Uy (mm)", "actual_Uy_mm"),
                        (a4, "Predicted Uy (mm)", "predicted_Uy_mm"),
                    ):
                        area.metric(label, f"{zero_small_displacements([row[column]])[0]:.5g}")
                ux_case, uy_case = st.tabs(["Case study Ux", "Case study Uy"])
                for direction, area in (("Ux", ux_case), ("Uy", uy_case)):
                    with area:
                        color = CASE_STUDY_COLORS.get(model_selected, "#333333")
                        st.markdown(f"### Peak |{direction}|")
                        st.plotly_chart(case_study_scatter(
                            sub, feature_selected, selected_simulation,
                            direction, color, stat_col), use_container_width=True,
                            key=f"shap_disp_{direction}")
                        st.plotly_chart(case_shap_scatter(
                            sub, feature_selected, selected_simulation,
                            direction, color, stat_col), use_container_width=True,
                            key=f"shap_phi_{direction}")
                        feature_rows = sample_subset.loc[
                            sample_subset["simulation_id"].eq(selected_simulation) &
                            sample_subset["output"].eq(direction)].copy()
                        feature_rows["Shapley contribution (mm)"] = (
                            feature_rows["shapley_value"] * 1000)
                        feature_rows = feature_rows.sort_values(
                            "Shapley contribution (mm)",
                            key=lambda col: col.abs(), ascending=False).head(15)
                        fig_bar = go.Figure(go.Bar(
                            x=feature_rows["Shapley contribution (mm)"],
                            y=feature_rows["feature"], orientation="h",
                            marker_color=color))
                        fig_bar.update_yaxes(autorange="reversed")
                        st.plotly_chart(case_plot_style(
                            fig_bar, "Shapley contribution (mm)", "Input feature",
                            f"Top 15 feature contributions: {selected_simulation} — {direction}"),
                            use_container_width=True, key=f"shap_bar_{direction}")
                st.download_button(
                    "Download selected simulation SHAP values",
                    sample_subset.loc[
                        sample_subset["simulation_id"].eq(selected_simulation)
                    ].to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_simulation}_sampled_shap.csv",
                    mime="text/csv", key="shap_case_download")
            except Exception as exc:
                st.error(f"Could not load the SHAP case study: {type(exc).__name__}: {exc}")
                st.caption("Ensure the SHAP CSV is shared with the same Google service account "
                           "as your prediction CSV.")
