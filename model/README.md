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

- `classes_ = [0, 1]`; **class 1 = stunting**. P(stunting) = `predict_proba[:, 1]`.
- The two z-scores drive ~85% of the model.

## ⚠️ Encoding assumption — VERIFY before trusting categorical inputs

The training notebook / `LabelEncoder` mapping was **not available**. The 0/1
meaning of the six binary features in `app.py:BINARY_FIELDS` is a **best guess**
(sklearn `LabelEncoder` alphabetical convention, cross-checked against the model's
learned risk direction where the signal was strong enough — e.g. `BB Lahir` 0 =
low birth weight matches the model).

Current assumed maps (form option value → meaning):

| Column | 0 | 1 |
|--------|---|---|
| `BB Lahir (gram)` | `< 2500 g (BBLR)` | `≥ 2500 g (Normal)` |
| `MEROKOK` | Tidak | Ya |
| `KEPEMILKAN JKN` | Tidak punya | Punya |
| `RIWAYAT DATANG POSYANDU` | Rutin | Tidak rutin |
| `STATUS KELIARGA` | Pra-sejahtera | Sejahtera |
| `POLA ASUH` | Baik | Kurang |

**To correct a map:** flip the `value` of the two `<option>`s for that field in
`app.py:BINARY_FIELDS`. Nothing else changes. The two z-scores are unaffected by
this uncertainty, so overall predictions remain meaningful.

## Runtime note (version skew)

The artifact was pickled with **scikit-learn 1.6.1** but `requirements.txt` pins
**1.9.0** (older sklearn has no wheels for Python 3.13/3.14). It unpickles under
1.9.0 with an `InconsistentVersionWarning`. For production, retrain on the pinned
version or pin scikit-learn to the training version and run on Python 3.12.
