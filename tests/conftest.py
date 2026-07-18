"""Shared pytest fixtures for the StuntingCare test suite.

Each test builds an isolated app from the factory with a test config and a
frozen clock, so redirects, session state, and the download report body are
deterministic. ``client`` uses the real trained model; ``no_model_client``
points at a missing artifact to exercise the rule-based fallback.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app import create_app
from config import Config

# Frozen clock -> recorded_at renders as "12 July 2026, 10:30".
FIXED_NOW = datetime(2026, 7, 12, 10, 30, 0)


class TestingConfig(Config):
    # Named "TestingConfig" (not "TestConfig") so pytest does not collect it.
    TESTING = True
    SECRET_KEY = "test-secret-key"


class NoModelConfig(TestingConfig):
    MODEL_PATH = "/nonexistent/no_model.joblib"


# A clearly underweight child -> high risk (STUNTING).
VALID_HIGH_RISK = {
    "child_name": "Budi",
    "bb_u": "-3.5",
    "z_score": "0",
    "birth_weight": "0",
    "merokok": "1",
    "jkn": "0",
    "riwayat_posyandu": "1",
    "status_keluarga": "0",
    "pola_asuh": "1",
}

# A healthy child -> low risk (TIDAK STUNTING).
VALID_LOW_RISK = {
    "child_name": "Aulia",
    "bb_u": "0.5",
    "z_score": "0.5",
    "birth_weight": "1",
    "merokok": "0",
    "jkn": "1",
    "riwayat_posyandu": "0",
    "status_keluarga": "1",
    "pola_asuh": "0",
}


@pytest.fixture
def app():
    application = create_app(TestingConfig)
    application.config["CLOCK"] = lambda: FIXED_NOW
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def no_model_client():
    application = create_app(NoModelConfig)
    application.config["CLOCK"] = lambda: FIXED_NOW
    return application.test_client()
