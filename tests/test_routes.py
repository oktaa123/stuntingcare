"""End-to-end checks for screening, reports, reset state, and validation."""
from __future__ import annotations

from time import perf_counter

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


def test_class_one_payload_renders_severely_stunting(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    page = client.get("/result")
    assert b'data-classification="SEVERELY STUNTING"' in page.data
    assert b'data-positive="true"' in page.data


def test_class_zero_payload_renders_stunting(client):
    client.post("/prediction", data=VALID_LOW_RISK)
    page = client.get("/result")
    assert b'data-classification="STUNTING"' in page.data
    assert b'data-positive="false"' in page.data


def test_result_uses_health_language_without_technical_terms(client):
    client.post("/prediction", data=VALID_LOW_RISK)
    page = client.get("/result").data.decode()
    assert "bukan diagnosis medis" in page
    assert "Probabilitas Prediksi" in page
    assert "Machine Learning" not in page
    assert "Random Forest" not in page
    assert ">AI<" not in page


def test_result_shows_only_factors_matching_the_submission(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    page = client.get("/result").data.decode()
    assert "Faktor yang Mempengaruhi Hasil Skrining" in page
    assert "Berat Badan Lahir" in page
    assert "Paparan Asap Rokok" in page
    assert "Kepemilikan JKN" in page
    assert "Kunjungan Posyandu" in page
    assert "Pola Asuh" in page
    # This input has a normal BB/TB z-score, so that factor must stay hidden.
    assert "Nilai Z-Score BB/TB" not in page


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


def test_download_is_a_print_ready_a4_pdf(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    resp = client.get("/result/download")
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"
    assert resp.headers["Content-Disposition"] == (
        'attachment; filename="SCR-20260712-0001-hasil-skrining.pdf"'
    )
    assert resp.data.startswith(b"%PDF-")
    assert b"/MediaBox [ 0 0 595.2756 841.8898 ]" in resp.data
    assert len(resp.data) > 10_000


def test_report_numbers_increment_for_repeated_screenings(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    with client.session_transaction() as screening_session:
        assert screening_session["prediction_result"]["report_number"] == "SCR-20260712-0001"

    client.get("/prediction")
    client.post("/prediction", data=VALID_LOW_RISK)
    with client.session_transaction() as screening_session:
        assert screening_session["prediction_result"]["report_number"] == "SCR-20260712-0002"


def test_screening_again_clears_result_session_and_form(client):
    client.post("/prediction", data=VALID_HIGH_RISK)
    form_page = client.get("/prediction")
    assert form_page.status_code == 200
    assert b'value="Budi"' not in form_page.data
    with client.session_transaction() as screening_session:
        assert "prediction_result" not in screening_session
    assert client.get("/result").headers["Location"].endswith("/prediction")


def test_dynamic_screening_pages_disable_browser_cache(client):
    response = client.get("/prediction")
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["Pragma"] == "no-cache"


def test_healthy_inputs_do_not_show_unrelated_factors_or_actions(client):
    client.post("/prediction", data=VALID_LOW_RISK)
    page = client.get("/result").data.decode()
    assert "Tidak ada faktor khusus yang perlu ditampilkan" in page
    assert "Balita terpapar asap rokok" not in page
    assert "Kunjungan pemantauan pertumbuhan ke Posyandu belum rutin" not in page
    assert "Jadikan rumah dan area bermain" not in page
    assert "Susun jadwal kunjungan Posyandu" not in page


def test_prediction_reuses_in_memory_model_and_completes_under_one_second(client, monkeypatch):
    monkeypatch.setattr("app.scoring.load_model", lambda _path: pytest.fail(
        "Model tidak boleh dimuat ulang saat prediksi"
    ))
    durations = []
    for index in range(3):
        payload = {**VALID_HIGH_RISK, "child_name": f"Budi {index}"}
        started = perf_counter()
        response = client.post("/prediction", data=payload)
        durations.append(perf_counter() - started)
        assert response.status_code == 302
    assert max(durations) < 1.0


def test_session_does_not_carry_raw_inputs(client):
    with client:
        client.post("/prediction", data=VALID_HIGH_RISK)
        from flask import session
        assert "inputs" not in session["prediction_result"]
