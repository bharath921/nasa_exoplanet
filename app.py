from hashlib import sha256

import pandas as pd
import streamlit as st

from exoplanet import (
    MAX_ROWS, default_features, demo_data, feature_columns, model_bytes, predict,
    read_csv, train,
)

st.set_page_config(page_title="Exoplanet Model Lab", page_icon="🪐", layout="wide")
st.title("🪐 Exoplanet Model Lab")
st.write("Explore a dataset, train a random forest, and compare its predictions with a simple baseline.")

source = st.radio("Dataset", ["Synthetic demo", "Upload CSV"], horizontal=True)
if source == "Synthetic demo":
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
target = st.selectbox(
    "Target column", frame.columns.tolist(),
    index=frame.columns.get_loc("pl_bmassj") if "pl_bmassj" in frame else None,
    key=f"target-{dataset_id}",
)
if target is None:
    st.info("Choose the column you want to predict.")
    st.stop()
available = feature_columns(frame, target)
if not available:
    st.warning("This target has no eligible numeric input columns.")
    st.stop()
st.caption("Use features you would actually know at prediction time. Exclude any value derived from the target.")
with st.form("train", clear_on_submit=False):
    features = st.multiselect(
        "Input features", available, default=default_features(available),
        key=f"features-{dataset_id}-{target}",
    )
    left, middle, right = st.columns(3)
    task = left.selectbox("Task", ["Regression", "Classification"],
                          index=int(target == "orbit_group"), key=f"task-{dataset_id}-{target}")
    trees = middle.slider("Trees", 25, 200, 100, step=25)
    test_percent = right.slider("Holdout size (%)", 15, 40, 25, step=5)
    submitted = st.form_submit_button("Train and evaluate", type="primary")

if submitted:
    st.session_state.pop("training_result", None)
    try:
        with st.spinner("Fitting and evaluating the random forest…"):
            artifact, report = train(
                frame, features, target, task.lower(), n_estimators=trees,
                test_size=test_percent / 100,
            )
            st.session_state.training_result = (dataset_id, artifact, report, model_bytes(artifact))
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get("training_result")
if not result or result[0] != dataset_id or result[1]["target"] != target:
    st.stop()

_, artifact, report, download = result
st.header("Holdout results")
st.caption(
    f"{artifact['task'].title()} for {target} · {report['train_rows']:,} training rows · "
    f"{report['test_rows']:,} test rows · {report['unlabeled_rows']:,} unusable targets · "
    f"{report['duplicate_rows']:,} duplicate examples removed"
)
for column, (name, value) in zip(st.columns(len(report["metrics"])), report["metrics"].items()):
    column.metric(name, "Unavailable" if value is None else f"{value:.3f}")

diagnostics, importance = st.tabs(["Model diagnostics", "Feature importance"])
with diagnostics:
    if artifact["task"] == "classification":
        st.write("Confusion matrix (rows: actual; columns: predicted)")
        st.dataframe(report["details"]["confusion"])
        with st.expander("Per-class scores"):
            st.dataframe(report["details"]["report"])
    else:
        st.write("Actual vs. predicted")
        st.scatter_chart(report["details"]["predictions"], x="Actual", y="Predicted")
with importance:
    st.caption("Random forest importance measures how the model uses each input.")
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
    if st.form_submit_button("Predict"):
        try:
            value = predict(artifact, pd.DataFrame([inputs]))[0]
        except ValueError:
            st.error("Could not predict from these values. Check the input ranges and units.")
        else:
            display = value if artifact["task"] == "classification" else format(value, ".6g")
            st.success(f"Predicted {artifact['target']}: {display}")
