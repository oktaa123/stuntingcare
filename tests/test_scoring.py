"""Unit tests for the scoring layer and the model-wiring contract."""
from __future__ import annotations

import pytest
import pandas as pd

from app import build_features, predict_risk
from config import Config
from utils import scoring
from tests.conftest import VALID_HIGH_RISK, VALID_LOW_RISK

MODEL = scoring.load_model(Config.MODEL_PATH)


def test_model_artifact_loads():
    assert MODEL is not None, "model/random_forest_model.joblib should be present"
    assert list(MODEL.feature_names_in_) == scoring.MODEL_FEATURES
    assert MODEL.n_features_in_ == 8
    assert list(MODEL.classes_) == [0, 1]
    assert scoring.MODEL_CLASS_LABELS == {
        0: "STUNTING",
        1: "SEVERELY STUNTING",
    }


def test_build_features_maps_to_model_columns():
    features = build_features(VALID_HIGH_RISK)
    assert set(features) == set(scoring.MODEL_FEATURES)
    assert features["BB/U"] == -3.5
    assert features["MEROKOK"] == 1  # int-encoded
    assert isinstance(features["MEROKOK"], int)


def test_class_one_maps_to_severely_stunting():
    result = predict_risk(VALID_HIGH_RISK, MODEL)
    assert result["source"] == "model"
    assert result["positive"] is True
    assert result["classification"] == "SEVERELY STUNTING"


def test_class_zero_maps_to_stunting():
    result = predict_risk(VALID_LOW_RISK, MODEL)
    assert result["positive"] is False
    assert result["classification"] == "STUNTING"


@pytest.mark.parametrize("payload", [VALID_HIGH_RISK, VALID_LOW_RISK])
def test_probability_within_bounds(payload):
    probability = predict_risk(payload, MODEL)["probability"]
    assert 0 <= probability <= 100


def test_prediction_and_probability_use_the_same_model_class():
    features = build_features(VALID_HIGH_RISK)
    prediction, probability = scoring.predict_class_and_probability(features, MODEL)
    frame = pd.DataFrame(
        [[features[name] for name in scoring.MODEL_FEATURES]],
        columns=scoring.MODEL_FEATURES,
    )
    direct_prediction = int(MODEL.predict(frame)[0])
    direct_probabilities = MODEL.predict_proba(frame)[0]
    direct_index = list(MODEL.classes_).index(direct_prediction)

    assert prediction == direct_prediction
    assert probability == pytest.approx(float(direct_probabilities[direct_index]))


MODEL_AUDIT_CASES = [
    (
        "normal",
        {**VALID_LOW_RISK, "bb_u": "-1.0", "z_score": "-1.2"},
    ),
    (
        "sedang",
        {**VALID_LOW_RISK, "bb_u": "-2.0", "z_score": "-2.3"},
    ),
    (
        "ekstrem",
        {
            **VALID_HIGH_RISK,
            "bb_u": "-3.8",
            "z_score": "-4.0",
        },
    ),
    ("kelas_satu", VALID_HIGH_RISK),
]


@pytest.mark.parametrize(("case_name", "payload"), MODEL_AUDIT_CASES)
def test_predict_risk_matches_direct_random_forest_output(case_name, payload):
    features = build_features(payload)
    frame = pd.DataFrame(
        [[features[name] for name in scoring.MODEL_FEATURES]],
        columns=scoring.MODEL_FEATURES,
    )
    direct_prediction = int(MODEL.predict(frame)[0])
    direct_probabilities = MODEL.predict_proba(frame)[0]
    direct_probability = float(
        direct_probabilities[list(MODEL.classes_).index(direct_prediction)]
    )

    result = predict_risk(payload, MODEL)

    assert result["classification"] == scoring.MODEL_CLASS_LABELS[direct_prediction], case_name
    assert result["probability"] == round(direct_probability * 100)


def test_probability_cannot_override_predict_category():
    class DeliberatelyContradictoryModel:
        classes_ = [0, 1]

        @staticmethod
        def predict(_frame):
            return [0]

        @staticmethod
        def predict_proba(_frame):
            return [[0.01, 0.99]]

    result = predict_risk(VALID_LOW_RISK, DeliberatelyContradictoryModel())
    assert result["classification"] == "STUNTING"
    assert result["probability"] == 1


def test_missing_model_is_rejected_instead_of_using_a_hardcoded_category():
    with pytest.raises(RuntimeError, match="Model Random Forest tidak tersedia"):
        predict_risk(VALID_HIGH_RISK, None)


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
    assert result["recorded_at"] == "02 Januari 2030, 03.04"
