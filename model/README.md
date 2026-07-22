# Model artifact

`random_forest_model.joblib` — a scikit-learn `RandomForestClassifier`
(280 trees) that predicts stunting risk. `config.Config.MODEL_PATH` points here.

## Feature schema (verified by introspecting the model)

The model was trained on a **pandas DataFrame**, so both the column **names** and
their **order** matter. `utils/scoring.py:MODEL_FEATURES` reproduces them exactly,
including the original header typos.

| # | Column (`feature_names_in_`) | Kind | Notes |
|---|------------------------------|------|-------|
| 1 | `BB/U`                       | float (z-score) | weight-for-age; ~41% importance |
| 2 | `Z Score BB/TB`             | float (z-score) | weight-for-height; ~44% importance |
| 3 | `MEROKOK`                   | binary 0/1 | ~3% |
| 4 | `KEPEMILKAN JKN` *(sic)*    | binary 0/1 | ~0.4% |
| 5 | `RIWAYAT DATANG POSYANDU`   | binary 0/1 | ~6.5% |
| 6 | `BB Lahir (gram)`           | binary 0/1 | **binarized in training (0 = low birth weight) — NOT raw grams**, despite the name; ~4% |
| 7 | `STATUS KELIARGA` *(sic)*   | binary 0/1 | ~0.8% |
| 8 | `POLA ASUH`                 | binary 0/1 | **0% importance — the model ignores it**, but the column is still required |

- `classes_ = [0, 1]`.
- The matching training notebook (`final_rf_=_tuning_.ipynb`) creates the target
  with `np.where(df['TB/U'] < -3, 1, 0)` and labels the confusion-matrix order
  as `["Stunting", "Severely"]`. The verified mapping is therefore:
  **0 = STUNTING, 1 = SEVERELY STUNTING**.
- The displayed probability is taken from the `predict_proba()` column matching
  the class returned by `model.predict()`; it never determines the category.
- The two z-scores drive ~85% of the model.

## Categorical input encoding

The matching notebook confirms that the categorical columns are represented as
0/1 and that `LabelEncoder` preserves those numeric codes. Separate encoder
artifacts containing the original human-readable meanings were not exported.
The existing form encodings below are therefore preserved; the output-label fix
does not alter preprocessing or any feature value.

Current assumed maps (form option value → meaning):

| Column | 0 | 1 |
|--------|---|---|
| `BB Lahir (gram)` | `< 2500 g (BBLR)` | `≥ 2500 g (Normal)` |
| `MEROKOK` | Tidak | Ya |
| `KEPEMILKAN JKN` | Tidak punya | Punya |
| `RIWAYAT DATANG POSYANDU` | Rutin | Tidak rutin |
| `STATUS KELIARGA` | Pra-sejahtera | Sejahtera |
| `POLA ASUH` | Baik | Kurang |

The two z-scores and all six categorical values are sent to the trained model in
the same names, order, and numeric representation used before this audit.

## Runtime note (version skew)

The artifact was pickled with **scikit-learn 1.6.1** but `requirements.txt` pins
**1.9.0** (older sklearn has no wheels for Python 3.13/3.14). It unpickles under
1.9.0 with an `InconsistentVersionWarning`. For production, retrain on the pinned
version or pin scikit-learn to the training version and run on Python 3.12.
