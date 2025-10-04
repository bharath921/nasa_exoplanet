Quick run and model artifact

This folder contains a small helper script `run_quick.py` that:

- reads a sample of `planets.csv` (2000 rows) to avoid memory issues,
- does minimal cleaning and selects a compact feature set,
- trains a small RandomForest (regression if `pl_bmassj` exists),
- saves the trained model to `rf_sample_model.joblib` in this folder.

To run (Windows PowerShell):

python "run_quick.py"

The saved model will be at:

./rf_sample_model.joblib

Install requirements if needed:

pip install -r requirements.txt
