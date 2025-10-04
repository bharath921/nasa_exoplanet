import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
import joblib
import tempfile

st.set_page_config(page_title='Simple RF Trainer', layout='wide')
st.title('Simple Random Forest Trainer')


def train_model_and_predict(df, features, target, n_estimators=100, random_state=42):
    
    data = df.copy()
    for c in ['hd_name', 'hip_name', 'tic_id', 'gaia_id']:
        if c in data.columns:
            try:
                data.drop(columns=c, inplace=True)
            except Exception:
                pass

    features = [f for f in features if f in data.columns]
    if not features:
        raise ValueError('No valid feature columns provided')

    X = data[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.median())

    y = None
    if target in data.columns:
        y = data[target].copy()

    is_classification = False
    labels = None
    if y is None:
       
        indicator_cols = [c for c in ['pl_name','pl_bmassj','pl_orbper','pl_rade','pl_radj'] if c in data.columns]
        if not indicator_cols:
            raise ValueError('No target provided and no indicator columns available to build is_planet label')
        y = data[indicator_cols].notnull().any(axis=1).astype(int)
        is_classification = True
    else:
        try:
            y = pd.to_numeric(y, errors='coerce')
            if y.isnull().all():
                y = data[target].astype('object')
                is_classification = True
            else:
                y = y.fillna(y.median())
                is_classification = False
        except Exception:
            is_classification = True

    if is_classification:
        if y.dtype == 'O' or y.dtype.name == 'category':
            labels, y_enc = np.unique(y, return_inverse=True)
            y_used = pd.Series(y_enc, index=y.index)
        else:
            labels = None
            y_used = y
    else:
        y_used = y

    stratify_arg = y_used if (hasattr(y_used, 'value_counts') and getattr(y_used, 'value_counts')().min() >= 2 if is_classification else False) else None
    if stratify_arg is None and is_classification:
        st.warning('Some classes have <2 samples; proceeding without stratify')

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_used, test_size=0.2, random_state=random_state, stratify=stratify_arg)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y_used, test_size=0.2, random_state=random_state)

    if is_classification:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    out = {'model': model, 'is_classification': is_classification, 'labels': labels, 'features': features}
    if is_classification:
        st.write('Accuracy:', round(accuracy_score(y_test, preds), 4))
        st.text('Classification report:')
        st.text(classification_report(y_test, preds))
    else:
        mse = mean_squared_error(y_test, preds)
        st.write('R2:', round(r2_score(y_test, preds), 4))
        st.write('RMSE:', round(np.sqrt(mse), 4))

    return out


uploaded_file = st.file_uploader('Upload CSV (cleaned preferred)', type=['csv'])
model_artifact = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    try:
        df = pd.read_csv(path, low_memory=True)
    except Exception:
        st.warning('Full read failed — reading a small sample instead')
        df = pd.read_csv(path, nrows=2000)

    st.write('Loaded dataset shape:', df.shape)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.sidebar.markdown('Select features and target')
    features = st.sidebar.multiselect('Features (numeric only recommended)', numeric_cols, default=numeric_cols[:5])
    target = st.sidebar.selectbox('Target column (leave blank to build is_planet)', [''] + df.columns.tolist())

    if st.sidebar.button('Train Random Forest'):
        try:
            chosen_target = target if target != '' else None
            res = train_model_and_predict(df, features, chosen_target, n_estimators=100)
            model_artifact = res
            model_path = 'rf_model.joblib'
            to_dump = {'model': res['model'], 'is_classification': res['is_classification'], 'labels': res['labels'], 'features': res['features']}
            joblib.dump(to_dump, model_path)
            st.success(f'Model + metadata saved to {model_path}')
            with open(model_path, 'rb') as f:
                st.download_button('Download model artifact', f, file_name=model_path)
        except Exception as e:
            st.error('Training failed: ' + str(e))

    if st.sidebar.button('Load saved model'):
        try:
            loaded = joblib.load('rf_model.joblib')
            model_artifact = loaded
            st.success('Loaded rf_model.joblib')
        except Exception as e:
            st.warning('Could not load rf_model.joblib: ' + str(e))

    st.markdown('---')
    st.subheader('Predict on a single example')
    if 'model' in (model_artifact or {}):
        ma = model_artifact
    else:
        try:
            ma = joblib.load('rf_model.joblib')
        except Exception:
            ma = None

    if ma is None:
        st.info('No trained model available yet. Train a model or load rf_model.joblib to enable single-row prediction.')
    else:
        feat_list = ma['features']
        inputs = {}
        cols = st.columns(len(feat_list)) if len(feat_list) <= 6 else None
        for i, f in enumerate(feat_list):
            if cols:
                inputs[f] = cols[i].number_input(f, value=float(df[f].median()) if f in df.columns else 0.0)
            else:
                inputs[f] = st.number_input(f, value=float(df[f].median()) if f in df.columns else 0.0)

        if st.button('Predict single row'):
            row = pd.DataFrame([inputs])
            for c in row.columns:
                row[c] = pd.to_numeric(row[c], errors='coerce')
            row = row.fillna(row.median())
            model = ma['model']
            is_clf = ma.get('is_classification', False)
            labels = ma.get('labels', None)
            pred = model.predict(row[ma['features']])
            if is_clf and labels is not None:
                try:
                    mapped = [labels[int(p)] for p in pred]
                    st.write('Predicted class:', mapped[0])
                except Exception:
                    st.write('Predicted numeric label:', pred[0])
            else:
                st.write('Predicted value:', float(pred[0]))

