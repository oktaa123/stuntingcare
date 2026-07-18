from __future__ import annotations

import os
from datetime import datetime

from flask import Flask, Response, redirect, render_template, request, session, url_for

from config import Config
from utils import scoring

# --- Form specification -------------------------------------------------------
# The form collects 8 model features. Two are continuous z-scores; the other six
# were binary {0, 1} in training, so they are single-choice selects whose option
# VALUES are the integer codes fed to the model.
#
# ⚠️ ENCODING ASSUMPTION: the training notebook / LabelEncoder map was not
# available, so the 0/1 meaning of the categorical features is a best guess
# (sklearn LabelEncoder's alphabetical convention, cross-checked against the
# model's learned risk direction where the signal was strong enough). The two
# z-scores drive ~85% of the model, so predictions stay meaningful, but VERIFY
# these maps against the training code before trusting the categorical inputs.
# See model/README.md. Flipping a map is a one-line change to a label's value.

NUMERIC_FIELDS = {
    "bb_u": {"column": "BB/U", "label": "BB/U (Z-score)", "min": -6, "max": 6},
    "z_score": {"column": "Z Score BB/TB", "label": "Z-Score BB/TB", "min": -6, "max": 6},
}

BINARY_FIELDS = {
    "birth_weight": {
        "column": "BB Lahir (gram)",
        "label": "Berat Badan Lahir",
        "options": [
            {"value": "0", "label": "< 2500 gram (BBLR)"},
            {"value": "1", "label": "≥ 2500 gram (Normal)"},
        ],
    },
    "merokok": {
        "column": "MEROKOK",
        "label": "Paparan Asap Rokok di Rumah",
        "options": [
            {"value": "0", "label": "Tidak"},
            {"value": "1", "label": "Ya"},
        ],
    },
    "jkn": {
        "column": "KEPEMILKAN JKN",
        "label": "Kepemilikan JKN",
        "options": [
            {"value": "0", "label": "Tidak Punya"},
            {"value": "1", "label": "Punya"},
        ],
    },
    "riwayat_posyandu": {
        "column": "RIWAYAT DATANG POSYANDU",
        "label": "Riwayat Datang ke Posyandu",
        "options": [
            {"value": "0", "label": "Rutin"},
            {"value": "1", "label": "Tidak Rutin"},
        ],
    },
    "status_keluarga": {
        "column": "STATUS KELIARGA",
        "label": "Status Keluarga",
        "options": [
            {"value": "0", "label": "Pra-Sejahtera"},
            {"value": "1", "label": "Sejahtera"},
        ],
    },
    "pola_asuh": {
        "column": "POLA ASUH",
        "label": "Pola Asuh",
        "options": [
            {"value": "0", "label": "Baik"},
            {"value": "1", "label": "Kurang"},
        ],
    },
}

FORM_FIELDS = ("child_name", *NUMERIC_FIELDS, *BINARY_FIELDS)

HIGH_RISK_RECOMMENDATIONS = [
    "Konsultasikan ke Puskesmas atau dokter untuk evaluasi lanjutan.",
    "Datang ke Posyandu setiap bulan untuk pemantauan pertumbuhan.",
    "Perbaiki pola makan dan pantau asupan gizi harian.",
    "Lakukan pengukuran ulang untuk memastikan kondisi anak.",
]

LOW_RISK_RECOMMENDATIONS = [
    "Pertahankan pola makan yang seimbang dan sesuai usia.",
    "Tetap kontrol rutin ke Posyandu atau fasilitas kesehatan.",
    "Pantau tinggi dan berat badan anak secara berkala.",
]


def validate_number(value: str, label: str, minimum: float, maximum: float, errors: list[str]) -> float | None:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError, AttributeError):
        errors.append(f"{label} harus berupa angka.")
        return None
    if not minimum <= number <= maximum:
        errors.append(f"{label} harus berada di antara {minimum:g} dan {maximum:g}.")
        return None
    return number


def build_features(form_data: dict[str, str]) -> dict[str, float]:
    """Map validated form values onto the model's exact column names."""
    features: dict[str, float] = {}
    for field, meta in NUMERIC_FIELDS.items():
        features[meta["column"]] = float(str(form_data[field]).replace(",", "."))
    for field, meta in BINARY_FIELDS.items():
        features[meta["column"]] = int(form_data[field])
    return features


def predict_risk(form_data: dict[str, str], model, now=datetime.now) -> dict:
    """Score one submission. Uses the trained model when available, else a
    transparent rule-based fallback. ``now`` is an injectable clock for tests."""
    features = build_features(form_data)

    if model is not None:
        probability = scoring.predict_probability(features, model)
        source = "model"
    else:
        probability = scoring.rule_based_probability(features)
        source = "rule_based"

    probability = max(0.01, min(0.99, probability))
    probability_percent = round(probability * 100)
    positive = probability >= 0.5

    return {
        "positive": positive,
        "classification": "STUNTING" if positive else "TIDAK STUNTING",
        "risk_level": "High Risk" if positive else "Low Risk",
        "probability": probability_percent,
        "source": source,
        "recommendations": HIGH_RISK_RECOMMENDATIONS if positive else LOW_RISK_RECOMMENDATIONS,
        "interpretation": (
            "Balita berisiko mengalami stunting. Disarankan pemeriksaan lebih lanjut di Posyandu atau Puskesmas."
            if positive else
            "Balita memiliki risiko stunting yang rendah. Tetap lakukan pemantauan pertumbuhan secara rutin."
        ),
        "recorded_at": now().strftime("%d %B %Y, %H:%M"),
        "child_name": form_data["child_name"],
    }


def create_app(config_object: type = Config) -> Flask:
    """Application factory so tests can build an isolated app with a test config."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Load the model once at startup; None when the artifact is missing.
    app.config.setdefault("MODEL", scoring.load_model(app.config["MODEL_PATH"]))

    def clock():
        return app.config.get("CLOCK", datetime.now)()

    def contributions():
        model = app.config["MODEL"]
        return scoring.feature_contributions(model) if model is not None else []

    @app.context_processor
    def inject_globals():
        return {"current_year": clock().year}

    @app.get("/")
    def home():
        return render_template("home.html", page_title="StuntingCare — Deteksi Dini, Tumbuh Optimal")

    @app.route("/prediction", methods=["GET", "POST"])
    def prediction():
        form_data = {field: "" for field in FORM_FIELDS}
        errors: list[str] = []
        if request.method == "POST":
            form_data = {field: request.form.get(field, "").strip() for field in FORM_FIELDS}

            if not form_data["child_name"]:
                errors.append("Nama balita wajib diisi.")

            for field, meta in NUMERIC_FIELDS.items():
                validate_number(form_data[field], meta["label"], meta["min"], meta["max"], errors)

            for field, meta in BINARY_FIELDS.items():
                allowed = {option["value"] for option in meta["options"]}
                if form_data[field] not in allowed:
                    errors.append(f"{meta['label']} wajib dipilih.")

            if not errors:
                session["prediction_result"] = predict_risk(form_data, app.config["MODEL"], now=clock)
                return redirect(url_for("result"))

        return render_template(
            "prediction.html",
            page_title="Prediksi Risiko — StuntingCare",
            form_data=form_data,
            errors=errors,
            numeric_fields=NUMERIC_FIELDS,
            binary_fields=BINARY_FIELDS,
        )

    @app.get("/result")
    def result():
        prediction_result = session.get("prediction_result")
        if not prediction_result:
            return redirect(url_for("prediction"))
        return render_template(
            "result.html",
            page_title="Hasil Prediksi — StuntingCare",
            result=prediction_result,
            contributions=contributions(),
        )

    @app.get("/result/download")
    def download_result():
        item = session.get("prediction_result")
        if not item:
            return redirect(url_for("prediction"))
        recommendations = "\n".join(f"- {line}" for line in item["recommendations"])
        report = (
            "STUNTINGCARE — HASIL PREDIKSI\n================================\n"
            f"Nama Balita : {item['child_name']}\nWaktu       : {item['recorded_at']}\n"
            f"Hasil       : {item['classification']}\nTingkat     : {item['risk_level']}\n"
            f"Probabilitas: {item['probability']}%\n\nInterpretasi\n{item['interpretation']}\n\n"
            f"Rekomendasi\n{recommendations}\n\n"
            "Catatan: Hasil ini adalah skrining awal dan bukan diagnosis medis.\n"
        )
        return Response(report, mimetype="text/plain", headers={"Content-Disposition": "attachment; filename=stuntingcare-result.txt"})

    @app.get("/about")
    def about():
        return render_template("about.html", page_title="Tentang Penelitian — StuntingCare")

    @app.get("/prediksi")
    def old_prediction():
        return redirect(url_for("prediction"), code=301)

    @app.get("/method")
    def method():
        return render_template("method.html", page_title="Metode Penelitian — StuntingCare")

    @app.get("/tentang")
    def old_about():
        return redirect(url_for("about"), code=301)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
