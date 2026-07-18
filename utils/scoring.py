"""Model loading and scoring for the stunting screening.

The trained artifact is a scikit-learn RandomForestClassifier saved with joblib.
It was trained on a pandas DataFrame with 8 columns, so column NAMES and ORDER
both matter — they are reproduced verbatim in ``MODEL_FEATURES`` below (including
the original typos in the training data's headers).

Probing the model shows:
  * ``BB/U`` and ``Z Score BB/TB`` are continuous z-scores (together ~85% of the
    model's importance).
  * the other 6 features were binary {0, 1} in training — including
    ``BB Lahir (gram)``, which was binarized (0 = low birth weight) rather than
    fed as raw grams, despite its name.
  * classes_ = [0, 1]; class 1 = stunting.
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

POSITIVE_CLASS = 1  # 1 = stunting

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


def predict_probability(features: dict[str, float], model) -> float:
    """P(stunting) from the model, using its exact column names + order.

    ``features`` must be keyed by the MODEL_FEATURES column names.
    """
    import pandas as pd

    frame = pd.DataFrame([[features[name] for name in MODEL_FEATURES]], columns=MODEL_FEATURES)
    proba = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    index = classes.index(POSITIVE_CLASS) if POSITIVE_CLASS in classes else len(classes) - 1
    return float(proba[index])


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


def rule_based_probability(features: dict[str, float]) -> float:
    """Transparent fallback used only when the model file is missing.

    Clinical convention: lower weight-for-age / weight-for-height z-scores mean
    higher stunting risk.
    """
    score = 0.05
    bb_u = features["BB/U"]
    z_bb_tb = features["Z Score BB/TB"]

    if bb_u < -3:
        score += 0.55
    elif bb_u < -2:
        score += 0.35
    elif bb_u < -1:
        score += 0.12

    if z_bb_tb < -3:
        score += 0.25
    elif z_bb_tb < -2:
        score += 0.15

    if features["BB Lahir (gram)"] == 0:  # 0 = low birth weight
        score += 0.08
    if features["RIWAYAT DATANG POSYANDU"] == 1:  # 1 = not routine
        score += 0.05

    return max(0.02, min(0.98, score))
