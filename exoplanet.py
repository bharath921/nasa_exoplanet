"""Shared CSV, training, and inference logic for the app and CLI."""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

MAX_ROWS = 50_000
IDENTIFIERS = {"pl_name", "hostname", "hd_name", "hip_name", "tic_id", "gaia_id", "gaia_dr3_id"}
PREFERRED = ("pl_orbper", "pl_rade", "sy_jmag", "sy_hmag", "sy_kmag", "st_mass", "st_rad")


def read_csv(source, limit=MAX_ROWS):
    """Read at most limit rows and report whether the file had more."""
    frame = pd.read_csv(source, comment="#", nrows=limit + 1, low_memory=False)
    if frame.empty:
        raise ValueError("The CSV must contain a header and at least one data row.")
    return frame.iloc[:limit], len(frame) > limit


def numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float).replace(
        [np.inf, -np.inf], np.nan
    )


def feature_columns(frame, target):
    excluded = IDENTIFIERS | {target, f"{target}err1", f"{target}err2", f"{target}lim"}
    # NASA tables may contain the target in both Earth and Jupiter units.
    for pair in ({"pl_bmassj", "pl_bmasse"}, {"pl_radj", "pl_rade"}, {"pl_massj", "pl_masse"}):
        if target in pair:
            for name in pair:
                excluded.update({name, f"{name}err1", f"{name}err2", f"{name}lim"})
    return [
        col for col in frame
        if col not in excluded and not col.startswith("Unnamed:") and numeric(frame[col]).notna().any()
    ]


def default_features(columns):
    return [col for col in PREFERRED if col in columns] or list(columns[:5])


def demo_data():
    """Deterministic synthetic data for trying the workflow, not scientific analysis."""
    rng = np.random.default_rng(42)
    period = np.exp(rng.uniform(-1, 6, 320))
    star_mass = rng.uniform(0.5, 1.6, len(period))
    radius = rng.uniform(0.8, 15, len(period))
    mass = np.maximum(0.001, 0.002 * radius ** 2.2 + 0.14 * star_mass
                      + 0.01 * np.log(period) + rng.normal(0, 0.05, len(period)))
    frame = pd.DataFrame({
        "pl_orbper": period, "st_mass": star_mass, "pl_rade": radius,
        "pl_bmassj": mass, "orbit_group": np.where(period < 30, "Short orbit", "Long orbit"),
    })
    frame.loc[::19, "st_mass"] = np.nan
    frame.loc[::31, "pl_bmassj"] = np.nan
    return frame


def train(frame, features, target, task, *, n_estimators=100, test_size=0.25, seed=42):
    features = list(features)
    if task not in {"regression", "classification"}:
        raise ValueError("Choose regression or classification.")
    if target not in frame:
        raise ValueError("Select a target column from the dataset.")
    if not features or len(features) != len(set(features)):
        raise ValueError("Select at least one feature, with no repeated columns.")
    allowed = set(feature_columns(frame, target))
    if invalid := set(features) - allowed:
        raise ValueError(f"Invalid or excluded features: {', '.join(sorted(invalid))}")
    if not 0 < test_size < 1:
        raise ValueError("The test fraction must be between zero and one.")

    X = frame[features].apply(numeric)
    if task == "classification":
        y = frame[target].astype("string").str.strip().replace("", pd.NA)
    else:
        y = numeric(frame[target])

    valid = y.notna()
    data = X.loc[valid].copy()
    label = object()
    data[label] = y.loc[valid]
    data = data.drop_duplicates()
    X, y = data[features], data[label].to_numpy(
        dtype=str if task == "classification" else float
    )
    if len(y) < 8:
        raise ValueError("Need at least eight distinct rows with usable target values.")
    n_test = int(np.ceil(test_size * len(y)))
    if min(n_test, len(y) - n_test) < 2:
        raise ValueError("Choose a split with at least two rows on each side.")

    classification = task == "classification"
    if classification:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification needs at least two classes.")
        if counts.min() < 2 or min(n_test, len(y) - n_test) < len(classes):
            raise ValueError("Each class needs examples on both sides of the split. Add data, combine rare classes, or adjust the test fraction.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if classification else None,
    )
    if classification and (len(np.unique(y_train)) != len(classes) or len(np.unique(y_test)) != len(classes)):
        raise ValueError("A class is absent from the training or test set. Add labeled examples or adjust the test fraction.")
    if empty := X_train.columns[X_train.isna().all()].tolist():
        raise ValueError(f"No usable training values for: {', '.join(empty)}. Remove those features.")

    forest = RandomForestClassifier if classification else RandomForestRegressor
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"), forest(n_estimators=n_estimators, random_state=seed)
    )
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    if classification:
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train).predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, predicted),
            "Balanced accuracy": balanced_accuracy_score(y_test, predicted),
            "Baseline balanced accuracy": balanced_accuracy_score(y_test, baseline),
        }
        details = {
            "report": pd.DataFrame(classification_report(
                y_test, predicted, output_dict=True, zero_division=0
            )).T,
            "confusion": pd.DataFrame(
                confusion_matrix(y_test, predicted, labels=classes),
                index=pd.Index(classes, name="Actual"),
                columns=pd.Index(classes, name="Predicted"),
            ),
        }
    else:
        baseline = DummyRegressor(strategy="median").fit(X_train, y_train).predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "R²": r2_score(y_test, predicted) if np.ptp(y_test) else None,
            "Baseline MAE": mean_absolute_error(y_test, baseline),
        }
        details = {"predictions": pd.DataFrame({"Actual": y_test, "Predicted": predicted})}

    artifact = {
        "format_version": 1, "sklearn_version": sklearn.__version__, "pipeline": pipeline,
        "task": task, "target": target, "features": features,
        "defaults": dict(zip(features, map(float, pipeline[0].statistics_))),
    }
    return artifact, {
        "metrics": metrics, "details": details, "train_rows": len(y_train),
        "test_rows": len(y_test), "unlabeled_rows": int((~valid).sum()),
        "duplicate_rows": int(valid.sum() - len(y)),
        "importance": pd.Series(pipeline[-1].feature_importances_, index=features)
        .sort_values(ascending=False),
    }


def predict(artifact, frame):
    features = artifact["features"]
    if missing := [col for col in features if col not in frame]:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")
    return artifact["pipeline"].predict(frame[features].apply(numeric))


def model_bytes(artifact):
    buffer = BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    return buffer.getvalue()
