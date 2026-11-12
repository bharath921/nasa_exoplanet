# Exoplanet Model Lab

Train a random forest on a labeled CSV, evaluate it on held-out data, and try predictions in a browser. No dataset is required to run the synthetic demo.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Choose **Synthetic demo** to try mass regression (`pl_bmassj`) or orbit group classification (`orbit_group`). Choose **Upload CSV** for your own data. Select a real target column, choose regression for a measured value or classification for labels (including numeric codes), and select inputs available when making a prediction. NASA-style CSV metadata lines beginning with `#` are skipped. The app accepts uploads up to 25 MB and uses up to the first 50,000 rows.

Choose **Kepler transit vetting** to classify already flagged transit signals as confirmed planets or false positives. It uses only fitted period (days), duration (hours), depth (ppm), and signal-to-noise. Star IDs, disposition scores, vetting flags, and unresolved candidates are excluded from the model. This evaluates catalog vetting within Kepler. Finding signals in raw light curves or testing on another survey requires separate evaluation.

The app compares Random Forest, Logistic Regression, Decision Tree, Support Vector Machine, K-Nearest Neighbors, and Histogram Gradient Boosting on the same five folds of the training data. The folds and the final holdout keep signals from the same Kepler star together. Each fold fits preprocessing from its own training rows. The model with the highest mean fold balanced accuracy is refit on all training stars and scored on the held-out stars.

The bundled snapshot comes from the [NASA Exoplanet Archive cumulative KOI table](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html), retrieved November 12, 2026. To reproduce the download, run this query through the [Archive TAP service](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html) and select CSV output.

```sql
select kepid, kepoi_name, koi_disposition, koi_period, koi_duration, koi_depth, koi_model_snr
from cumulative
where koi_disposition in ('CONFIRMED', 'FALSE POSITIVE')
order by kepoi_name
```

The app drops rows with missing or invalid targets and removes duplicate feature/target examples. It splits the remaining data before fitting the feature imputer or model. Classification uses a stratified holdout and reports balanced accuracy alongside a majority-class baseline; regression compares MAE with a median-prediction baseline. The model download includes the fitted imputer, feature order, target, and task.

A catalog of confirmed planets cannot measure planet detection accuracy: it needs independently labeled positive and negative examples. Check inputs for target proxies, and treat the random holdout as exploratory. If multiple observations share a planet or host star, or you need to predict future discoveries, use a group or time-based evaluation before drawing scientific conclusions.

### Command line and notebook

```powershell
python run_quick.py planets.csv --target pl_bmassj --task regression --features pl_orbper sy_jmag sy_hmag --output mass_model.joblib
```

The CLI accepts any local CSV path and refuses to overwrite an existing output. The notebook follows the same workflow and starts with the synthetic data; set its CSV path to use a catalog instead. This project does not bundle a real catalog.

The two original `.joblib` files are legacy artifacts. The app does not load them; retrain to get the current format. For offline inference, load only artifacts you trust with the same dependency versions used for training, then call `exoplanet.predict(artifact, dataframe)`.

Run the checks with `python -m unittest discover -s tests`.
