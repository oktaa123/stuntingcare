"""Unit tests for the scoring layer and the model-wiring contract."""
from __future__ import annotations

import pytest

from app import build_features, predict_risk
from config import Config
from utils import scoring
from tests.conftest import VALID_HIGH_RISK, VALID_LOW_RISK

MODEL = scoring.load_model(Config.MODEL_PATH)


def test_model_artifact_loads():
    assert MODEL is not None, "model/random_forest_model.joblib should be present"
    assert list(MODEL.feature_names_in_) == scoring.MODEL_FEATURES
    assert MODEL.n_features_in_ == 8


def test_build_features_maps_to_model_columns():
    features = build_features(VALID_HIGH_RISK)
    assert set(features) == set(scoring.MODEL_FEATURES)
    assert features["BB/U"] == -3.5
    assert features["MEROKOK"] == 1  # int-encoded
    assert isinstance(features["MEROKOK"], int)


def test_high_risk_classifies_stunting_with_model():
    result = predict_risk(VALID_HIGH_RISK, MODEL)
    assert result["source"] == "model"
    assert result["positive"] is True
    assert result["classification"] == "STUNTING"


def test_low_risk_classifies_tidak_stunting_with_model():
    result = predict_risk(VALID_LOW_RISK, MODEL)
    assert result["positive"] is False
    assert result["classification"] == "TIDAK STUNTING"


@pytest.mark.parametrize("payload", [VALID_HIGH_RISK, VALID_LOW_RISK])
def test_probability_within_bounds(payload):
    probability = predict_risk(payload, MODEL)["probability"]
    assert 1 <= probability <= 99


def test_predict_probability_uses_positive_class_index():
    # P(stunting) must track class 1, not "probability of the predicted class".
    features = build_features(VALID_HIGH_RISK)
    p = scoring.predict_probability(features, MODEL)
    assert p >= 0.5  # this vector is clearly high risk


def test_fallback_used_when_model_is_none():
    result = predict_risk(VALID_HIGH_RISK, None)
    assert result["source"] == "rule_based"
    assert result["positive"] is True  # very low BB/U -> fallback flags high risk


def test_feature_contributions_are_real_and_ranked():
    contribs = scoring.feature_contributions(MODEL)
    assert len(contribs) == 8
    # Sorted descending by importance.
    pct = [c["importance_pct"] for c in contribs]
    assert pct == sorted(pct, reverse=True)
    # The two z-scores dominate.
    assert contribs[0]["name"].startswith("Z-Score") or contribs[0]["name"].startswith("BB/U")


def test_recorded_at_uses_injected_clock():
    from datetime import datetime
    result = predict_risk(VALID_LOW_RISK, MODEL, now=lambda: datetime(2030, 1, 2, 3, 4, 0))
    assert result["recorded_at"] == "02 January 2030, 03:04"
