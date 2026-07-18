"""End-to-end route tests driving the app through Flask's test client.

Covers every route and state: static pages, legacy 301 redirects, both branches
of validation, both session states for /result and the download, the exact
download body under a frozen clock, and the model vs rule-based source flag.
"""
from __future__ import annotations

import pytest

from tests.conftest import VALID_HIGH_RISK, VALID_LOW_RISK


# --- Static / navigational pages -------------------------------------------------

def test_home_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"StuntingCare" in resp.data


def test_prediction_form_ok(client):
    resp = client.get("/prediction")
    assert resp.status_code == 200
    assert b"data-prediction-form" in resp.data
    # The 8 model inputs are present.
    for field in ("bb_u", "z_score", "birth_weight", "merokok", "jkn",
                  "riwayat_posyandu", "status_keluarga", "pola_asuh"):
        assert f'name="{field}"'.encode() in resp.data


def test_about_ok(client):
    assert client.get("/about").status_code == 200


def test_method_ok(client):
    assert client.get("/method").status_code == 200


@pytest.mark.parametrize("path,target", [("/prediksi", "/prediction"), ("/tentang", "/about")])
def test_legacy_redirects(client, path, target):
    resp = client.get(path)
    assert resp.status_code == 301
    assert resp.headers["Location"].endswith(target)


# --- Successful submission -------------------------------------------------------

def test_post_valid_redirects_to_result_and_sets_session(client):
    resp = client.post("/prediction", data=VALID_HIGH_RISK)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/result")
    result_page = client.get("/result")
    assert result_page.status_code == 200
    assert b'data-testid="result-classification"' in result_page.data


def test_high_risk_payload_renders_stunting(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    page = client.get("/result")
    assert b'data-classification="STUNTING"' in page.data
    assert b'data-positive="true"' in page.data


def test_low_risk_payload_renders_tidak_stunting(client):
    client.post("/prediction", data=VALID_LOW_RISK)
    page = client.get("/result")
    assert b'data-classification="TIDAK STUNTING"' in page.data
    assert b'data-positive="false"' in page.data


def test_result_uses_model_source_disclaimer(client):
    client.post("/prediction", data=VALID_LOW_RISK)
    page = client.get("/result").data.decode()
    assert "model machine learning (Random Forest)" in page


def test_result_shows_real_importance_card(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    page = client.get("/result").data.decode()
    assert "Faktor Paling Berpengaruh" in page
    assert "Z-Score BB/TB" in page  # top feature by importance


# --- Rule-based fallback (model missing) ----------------------------------------

def test_fallback_source_when_model_missing(no_model_client):
    no_model_client.post("/prediction", data=VALID_HIGH_RISK)
    page = no_model_client.get("/result").data.decode()
    assert "berbasis aturan (model belum aktif)" in page
    # No importance card without a model.
    assert "Faktor Paling Berpengaruh" not in page


# --- Validation failures ---------------------------------------------------------

def test_missing_child_name_re_renders_with_error(client):
    resp = client.post("/prediction", data={**VALID_HIGH_RISK, "child_name": ""})
    assert resp.status_code == 200
    assert b"Nama balita wajib diisi." in resp.data
    assert b'data-testid="form-errors"' in resp.data


def test_missing_binary_choice_re_renders_with_error(client):
    resp = client.post("/prediction", data={**VALID_HIGH_RISK, "merokok": ""})
    assert resp.status_code == 200
    assert "Paparan Asap Rokok".encode() in resp.data


def test_bogus_binary_value_is_rejected(client):
    # Unlike the old form, out-of-set select values are now rejected.
    resp = client.post("/prediction", data={**VALID_HIGH_RISK, "jkn": "9"})
    assert resp.status_code == 200
    assert b"Kepemilikan JKN wajib dipilih." in resp.data


def test_non_numeric_zscore_re_renders_with_error(client):
    resp = client.post("/prediction", data={**VALID_HIGH_RISK, "bb_u": "abc"})
    assert resp.status_code == 200
    assert b"harus berupa angka" in resp.data


def test_out_of_range_zscore_re_renders_with_error(client):
    resp = client.post("/prediction", data={**VALID_HIGH_RISK, "z_score": "50"})
    assert resp.status_code == 200
    assert b"harus berada di antara" in resp.data


def test_validation_failure_does_not_set_session(client):
    client.post("/prediction", data={**VALID_HIGH_RISK, "child_name": ""})
    resp = client.get("/result")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/prediction")


# --- /result and /result/download session states --------------------------------

def test_result_without_session_redirects_to_form(client):
    resp = client.get("/result")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/prediction")


def test_download_without_session_redirects_to_form(client):
    resp = client.get("/result/download")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/prediction")


def test_download_report_body_is_deterministic(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    resp = client.get("/result/download")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    assert resp.headers["Content-Disposition"] == "attachment; filename=stuntingcare-result.txt"
    body = resp.data.decode()
    assert "Nama Balita : Budi" in body
    assert "Hasil       : STUNTING" in body
    assert "12 July 2026, 10:30" in body  # frozen clock


def test_session_does_not_carry_raw_inputs(client):
    with client:
        client.post("/prediction", data=VALID_HIGH_RISK)
        from flask import session
        assert "inputs" not in session["prediction_result"]
