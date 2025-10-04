import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib

CSV_PATH = r'c:\Users\Dell\OneDrive\Desktop\ml\project nasa\planets.csv'

try:
    df = pd.read_csv(CSV_PATH, nrows=2000)
except Exception as e:
    print('Failed to read sample from CSV:', e)
    raise

print('Read sample shape:', df.shape)
print(df.columns[:20])
df = df.dropna(axis=1, how='all')
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[c].isnull().any():
        df[c].fillna(df[c].median(), inplace=True)
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].fillna('Unknown')

sel = [c for c in ['pl_orbper','sy_jmag','sy_hmag','sy_kmag','disc_year'] if c in df.columns]
print('Selected features:', sel)
if not sel:
    sel = list(df.select_dtypes(include=[np.number]).columns[:5])
    print('Fallback numeric features:', sel)

if 'pl_bmassj' in df.columns:
    tar = 'pl_bmassj'
    is_regression = True
else:
    tar = None
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].nunique() <= 10:
            tar = c
            is_regression = False
            break
    if tar is None:
        print('No suitable target found in sample. Exiting.')
        raise SystemExit(1)

print('Target:', tar, 'regression?', is_regression)

X = df[sel]
y = df[tar]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if is_regression:
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('R2:', r2_score(y_test, y_pred))
    out_path = r'c:\Users\Dell\OneDrive\Desktop\ml\project nasa\\rf_sample_model.joblib'
    joblib.dump(model, out_path)
    print('Saved trained model to', out_path)
else:
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    if y_train.dtype == 'O' or y_train.dtype.name == 'category':
        y_train = pd.factorize(y_train)[0]
        y_test = pd.factorize(y_test)[0]
    try:
        unq, cnt = np.unique(y_train, return_counts=True)
        for u, c in zip(unq, cnt):
            print(f"Class {u} contains {c} samples")
    except Exception as e:
        print('Could not compute class counts:', e)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    out_path = r'c:\Users\Dell\OneDrive\Desktop\ml\project nasa\\rf_sample_model.joblib'
    joblib.dump(model, out_path)
    print('Saved trained model to', out_path)

print('Done')
