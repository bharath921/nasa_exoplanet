from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from exoplanet import (
    KEPLER_FEATURES, KEPLER_TARGET, MAX_ROWS, default_features, demo_data, feature_columns, model_bytes, predict,
    read_csv, similar_kepler_signals, train,
)

st.set_page_config(page_title="Exoplanet Model Lab", page_icon="🪐", layout="wide")
st.title("🪐 Exoplanet Model Lab")
st.write("Train a model on a dataset and compare its predictions with a simple baseline.")

source = st.radio("Dataset", ["Synthetic demo", "Kepler transit vetting", "Upload CSV"], horizontal=True)
kepler = source == "Kepler transit vetting"
if kepler:
    st.info("Classify previously flagged Kepler transit signals as confirmed planets or false positives.")
    st.caption("Source: NASA Exoplanet Archive. Compare six classifiers with five-fold cross-validation grouped by star.")
    dataset_id = "kepler"
    frame, truncated = read_csv(Path(__file__).parent / "data" / "kepler_koi.csv")
elif source == "Synthetic demo":
    st.info("The demo uses simulated values. Its scores do not describe real exoplanets.")
    dataset_id, frame, truncated = "demo", demo_data(), False
else:
    uploaded = st.file_uploader("Exoplanet catalog or labeled dataset", type="csv")
    if uploaded is None:
        st.stop()
    dataset_id = sha256(uploaded.getbuffer()).hexdigest()
    try:
        if st.session_state.get("csv_id") != dataset_id:
            st.session_state.csv_data = read_csv(uploaded)
            st.session_state.csv_id = dataset_id
        frame, truncated = st.session_state.csv_data
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

if truncated:
    st.warning(f"Using the first {MAX_ROWS:,} rows. If the catalog is sorted, upload a representative sample.")

with st.expander(f"Dataset preview · {len(frame):,} rows × {len(frame.columns)} columns", expanded=True):
    st.dataframe(frame.head(10), hide_index=True, width="stretch")

st.header("Train a model")
target = KEPLER_TARGET if kepler else st.selectbox(
    "Target column", frame.columns.tolist(),
    index=frame.columns.get_loc("pl_bmassj") if "pl_bmassj" in frame else None,
    key=f"target-{dataset_id}",
)
if target is None:
    st.info("Choose the column you want to predict.")
    st.stop()
available = KEPLER_FEATURES if kepler else feature_columns(frame, target)
if not available:
    st.warning("This target has no eligible numeric input columns.")
    st.stop()
st.caption("Use features you would actually know at prediction time. Exclude any value derived from the target.")
with st.form("train", clear_on_submit=False):
    features = st.multiselect(
        "Input features", available, default=KEPLER_FEATURES if kepler else default_features(available),
        key=f"features-{dataset_id}-{target}", disabled=kepler,
    )
    left, middle, right = st.columns(3)
    task = left.selectbox("Task", ["Regression", "Classification"],
                          index=int(kepler or target == "orbit_group"), key=f"task-{dataset_id}-{target}",
                          disabled=kepler)
    trees = middle.slider("Random Forest trees" if kepler else "Trees", 25, 200, 100, step=25)
    test_percent = right.slider("Holdout stars (%)" if kepler else "Holdout rows (%)", 15, 40, 25, step=5)
    submitted = st.form_submit_button("Train and evaluate", type="primary")

if submitted:
    st.session_state.pop("training_result", None)
    try:
        with st.spinner("Comparing classifiers across five folds…" if kepler else "Fitting and evaluating the random forest…"):
            artifact, report = train(
                frame, features, target, task.lower(), n_estimators=trees,
                test_size=test_percent / 100, group_column="kepid" if kepler else None,
                compare_models=kepler,
            )
            st.session_state.training_result = (dataset_id, artifact, report, model_bytes(artifact))
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get("training_result")
if not result or result[0] != dataset_id or result[1]["target"] != target:
    st.stop()

_, artifact, report, download = result
if kepler:
    st.header("Cross-validation results")
    st.caption("The highest mean balanced accuracy across five training folds selects the model. Test stars are excluded from every fold.")
    cv_scores = report["cross_validation"]
    st.dataframe(cv_scores.style.format("{:.2%}", subset=cv_scores.columns[1:]),
                 hide_index=True, width="stretch")
st.header("Holdout results")
if kepler:
    st.caption(f"{report['train_groups']:,} training stars · {report['test_groups']:,} test stars")
st.caption(
    f"{artifact['model_name']} · {artifact['task'].title()} for {target} · {report['train_rows']:,} training rows · "
    f"{report['test_rows']:,} test rows · {report['unlabeled_rows']:,} unusable targets · "
    f"{report['duplicate_rows']:,} duplicate examples removed"
)
for column, (name, value) in zip(st.columns(len(report["metrics"])), report["metrics"].items()):
    column.metric(name, "Unavailable" if value is None else f"{value:.3f}")

tabs = ["Model diagnostics"]
if report["importance"] is not None:
    tabs.append("Feature importance")
diagnostics, *importance = st.tabs(tabs)
with diagnostics:
    if artifact["task"] == "classification":
        st.write("Confusion matrix (rows: actual; columns: predicted)")
        st.dataframe(report["details"]["confusion"])
        with st.expander("Per-class scores"):
            st.dataframe(report["details"]["report"])
    else:
        st.write("Actual vs. predicted")
        st.scatter_chart(report["details"]["predictions"], x="Actual", y="Predicted")
if importance:
    with importance[0]:
        st.caption("Tree feature importance measures how the fitted model uses each input.")
        st.bar_chart(report["importance"].rename("Importance"), horizontal=True)

st.download_button(
    "Download trained pipeline", download, "exoplanet_model.joblib",
    mime="application/octet-stream",
)

st.header("Try a prediction")
st.caption("Inputs start at their training-set medians. Use the same units as your dataset.")
with st.form("predict"):
    inputs = {
        col: st.number_input(col, value=artifact["defaults"][col], format="%.6g",
                             key=f"predict-{dataset_id}-{target}-{col}-{artifact['defaults'][col]}")
        for col in artifact["features"]
    }
    predict_clicked = st.form_submit_button("Predict")

if predict_clicked:
    if kepler and any(value <= 0 for value in inputs.values()):
        st.error("Enter a positive value for each transit measurement.")
        st.stop()
    try:
        value = predict(artifact, pd.DataFrame([inputs]))[0]
    except ValueError:
        st.error("Could not predict from these values. Check the input ranges and units.")
    else:
        display = value if artifact["task"] == "classification" else format(value, ".6g")
        st.success(f"Predicted {artifact['target']}: {display}")
        if kepler:
            st.subheader("Similar Kepler signals")
            st.caption("Closest catalog examples from the training stars, based on all four measurements. The labels shown are from the catalog.")
            measurements = {
                "koi_period": "Period (days)", "koi_duration": "Duration (hours)",
                "koi_depth": "Depth (ppm)", "koi_model_snr": "Signal-to-noise",
            }
            neighbors = similar_kepler_signals(frame.loc[report["training_index"]], inputs)
            if neighbors.empty:
                st.info("No comparable signals are available.")
            else:
                query = pd.DataFrame([{"kepoi_name": "Your signal", KEPLER_TARGET: "Unknown", **inputs}])
                comparison = pd.concat([query, neighbors], ignore_index=True)
                st.dataframe(
                    comparison.rename(columns={
                        "kepoi_name": "Signal", KEPLER_TARGET: "Catalog label", **measurements
                    }),
                    hide_index=True, width="stretch",
                )
