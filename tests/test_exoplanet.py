import io
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from exoplanet import demo_data, feature_columns, model_bytes, predict, read_csv, train


class TrainingTests(unittest.TestCase):
    def test_imputer_ignores_holdout_and_unlabeled_rows(self):
        frame = pd.DataFrame({"x": np.arange(21, dtype=float), "target": np.arange(21, dtype=float)})
        train_indices, test_indices = train_test_split(np.arange(20), test_size=0.25, random_state=42)
        frame.loc[test_indices, "x"] += 1000
        frame.loc[20, ["x", "target"]] = [1e8, np.nan]
        artifact, report = train(frame, ["x"], "target", "regression", n_estimators=5)
        self.assertEqual(artifact["defaults"]["x"], np.median(frame.loc[train_indices, "x"]))
        self.assertEqual(report["unlabeled_rows"], 1)
        self.assertEqual(report["test_rows"], 5)

    def test_target_columns_cannot_be_features(self):
        frame = demo_data().assign(pl_bmasse=1, pl_bmassjerr1=1)
        for feature in ("pl_bmassj", "pl_bmasse", "pl_bmassjerr1"):
            self.assertNotIn(feature, feature_columns(frame, "pl_bmassj"))
            with self.assertRaisesRegex(ValueError, "excluded features"):
                train(frame, [feature], "pl_bmassj", "regression")

    def test_numeric_classes_and_pipeline_round_trip(self):
        frame = pd.DataFrame({"x": np.arange(40), "label": np.repeat([0, 1], 20)})
        artifact, report = train(frame, ["x"], "label", "classification", n_estimators=10)
        self.assertGreater(report["metrics"]["Balanced accuracy"], 0.75)
        restored = joblib.load(io.BytesIO(model_bytes(artifact)))
        inputs = pd.DataFrame({"x": [2, 35, np.nan]})
        np.testing.assert_array_equal(predict(restored, inputs), predict(artifact, inputs))
        self.assertEqual(predict(restored, inputs)[:2].tolist(), ["0", "1"])
        with self.assertRaisesRegex(ValueError, "Missing prediction columns"):
            predict(restored, pd.DataFrame({"y": [2]}))

    def test_rare_classes_fail_instead_of_silently_using_random_split(self):
        frame = pd.DataFrame({"x": range(20), "label": ["common"] * 19 + ["rare"]})
        with self.assertRaisesRegex(ValueError, "Each class needs examples"):
            train(frame, ["x"], "label", "classification")

    def test_nasa_header_and_bounded_read(self):
        frame, truncated = read_csv(io.StringIO("# catalog notes\nx,y\n1,2\n3,4\n"), limit=1)
        self.assertTrue(truncated)
        self.assertEqual(frame.to_dict("records"), [{"x": 1, "y": 2}])


class AppTests(unittest.TestCase):
    def test_demo_trains_and_predicts_across_reruns(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("RMSE", [metric.label for metric in app.metric])
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success[0].value.startswith("Predicted pl_bmassj:"))
        app.selectbox[0].select("orbit_group").run()
        self.assertFalse(app.metric)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Balanced accuracy", [metric.label for metric in app.metric])
