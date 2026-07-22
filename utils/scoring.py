"""Model loading and scoring for the stunting screening.

The trained artifact is a scikit-learn RandomForestClassifier saved with joblib.
It was trained on a pandas DataFrame with 8 columns, so column NAMES and ORDER
both matter — they are reproduced verbatim in ``MODEL_FEATURES`` below (including
the original typos in the training data's headers).

The matching training notebook (``final_rf_=_tuning_.ipynb``) defines
``Target = np.where(TB/U < -3, 1, 0)`` and labels the confusion-matrix class
order as ``["Stunting", "Severely"]``. Therefore the trained target mapping is
0 = STUNTING and 1 = SEVERELY STUNTING.

Probing the model shows:
  * ``BB/U`` and ``Z Score BB/TB`` are continuous z-scores (together ~85% of the
    model's importance).
  * the other 6 features were binary {0, 1} in training — including
    ``BB Lahir (gram)``, which was binarized (0 = low birth weight) rather than
    fed as raw grams, despite its name.
  * classes_ = [0, 1], matching the notebook target mapping above.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Exact column names + order from model.feature_names_in_. Do not "fix" the
# spelling — the model matches columns by these strings.
MODEL_FEATURES: list[str] = [
    "BB/U",
    "Z Score BB/TB",
    "MEROKOK",
    "KEPEMILKAN JKN",
    "RIWAYAT DATANG POSYANDU",
    "BB Lahir (gram)",
    "STATUS KELIARGA",
    "POLA ASUH",
]

MODEL_CLASS_LABELS = {
    0: "STUNTING",
    1: "SEVERELY STUNTING",
}
SEVERE_CLASS = 1

# Human-friendly labels for the UI (the raw column names carry typos).
DISPLAY_NAMES = {
    "BB/U": "BB/U (berat badan menurut umur)",
    "Z Score BB/TB": "Z-Score BB/TB",
    "MEROKOK": "Paparan asap rokok",
    "KEPEMILKAN JKN": "Kepemilikan JKN",
    "RIWAYAT DATANG POSYANDU": "Rutin ke Posyandu",
    "BB Lahir (gram)": "Berat badan lahir",
    "STATUS KELIARGA": "Status keluarga",
    "POLA ASUH": "Pola asuh",
}

_model_cache: dict[str, Any] = {}


def load_model(model_path: str):
    """Load and cache the model. Returns None when the file is missing.

    A missing file is intentionally NOT cached, so a model dropped in after the
    process starts is still picked up on the next call.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]
    path = Path(model_path)
    if not path.exists():
        return None
    import joblib

    model = joblib.load(path)
    _model_cache[model_path] = model
    return model


def predict_class_and_probability(features: dict[str, float], model) -> tuple[int, float]:
    """Return ``model.predict`` and the probability of that predicted class.

    The category is always sourced from ``model.predict``. ``predict_proba`` is
    only used to obtain the display percentage for that same class.
    """
    import pandas as pd

    frame = pd.DataFrame([[features[name] for name in MODEL_FEATURES]], columns=MODEL_FEATURES)
    prediction = model.predict(frame)[0]
    prediction = int(prediction.item() if hasattr(prediction, "item") else prediction)
    classes = [int(value.item() if hasattr(value, "item") else value) for value in model.classes_]
    if set(classes) != set(MODEL_CLASS_LABELS):
        raise ValueError(
            f"Model classes {classes!r} do not match the verified label mapping "
            f"{sorted(MODEL_CLASS_LABELS)!r}."
        )
    if prediction not in classes:
        raise ValueError(f"Predicted class {prediction!r} is absent from model.classes_.")
    probability = float(model.predict_proba(frame)[0][classes.index(prediction)])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Model returned invalid probability {probability!r}.")
    return prediction, probability


def classification_for_prediction(prediction: int) -> str:
    """Map a numeric training target to its verified human-readable label."""
    try:
        return MODEL_CLASS_LABELS[int(prediction)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"No verified label mapping for model class {prediction!r}.") from error


def feature_contributions(model, limit: int | None = None) -> list[dict]:
    """Real, normalized feature_importances_ from the model.

    ``value`` is scaled 0..100 relative to the top feature (for the bar widths);
    ``importance_pct`` is the raw importance as a percentage.
    """
    importances = list(getattr(model, "feature_importances_", []))
    names = list(getattr(model, "feature_names_in_", MODEL_FEATURES))
    pairs = sorted(zip(names, importances), key=lambda kv: kv[1], reverse=True)
    if limit:
        pairs = pairs[:limit]
    top = pairs[0][1] if pairs and pairs[0][1] > 0 else 1.0
    return [
        {
            "name": DISPLAY_NAMES.get(name, name),
            "value": int(round(float(importance) / float(top) * 100)),
            "importance_pct": round(float(importance) * 100, 1),
        }
        for name, importance in pairs
    ]
