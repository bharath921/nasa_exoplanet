# Exoplanet Model Lab

Train a random forest on a labeled CSV, evaluate it on held-out data, and try predictions in a browser. No dataset is required to run the synthetic demo.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Choose **Synthetic demo** to try mass regression (`pl_bmassj`) or orbit group classification (`orbit_group`). Choose **Upload CSV** for your own data. Select a real target column, choose regression for a measured value or classification for labels (including numeric codes), and select inputs available when making a prediction. NASA-style CSV metadata lines beginning with `#` are skipped. The app accepts uploads up to 25 MB and uses up to the first 50,000 rows.

The app drops rows with missing or invalid targets and removes duplicate feature/target examples. It splits the remaining data before fitting the feature imputer or model. Classification uses a stratified holdout and reports balanced accuracy alongside a majority-class baseline; regression compares MAE with a median-prediction baseline. The model download includes the fitted imputer, feature order, target, and task.

A catalog of confirmed planets cannot measure planet detection accuracy: it needs independently labeled positive and negative examples. Check inputs for target proxies, and treat the random holdout as exploratory. If multiple observations share a planet or host star, or you need to predict future discoveries, use a group or time-based evaluation before drawing scientific conclusions.

### Command line and notebook

```powershell
python run_quick.py planets.csv --target pl_bmassj --task regression --features pl_orbper sy_jmag sy_hmag --output mass_model.joblib
```

The CLI accepts any local CSV path and refuses to overwrite an existing output. The notebook follows the same workflow and starts with the synthetic data; set its CSV path to use a catalog instead. This project does not bundle a real catalog.

The two original `.joblib` files are legacy artifacts. The app does not load them; retrain to get the current format. For offline inference, load only artifacts you trust with the same dependency versions used for training, then call `exoplanet.predict(artifact, dataframe)`.

Run the checks with `python -m unittest discover -s tests`.
