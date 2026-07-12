from __future__ import annotations

from datetime import datetime

from flask import Flask, Response, redirect, render_template, request, session, url_for

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

model = joblib.load("model/random_forest_model.joblib")

FEATURE_CONTRIBUTIONS = [
    {"name": "Berat Badan Lahir", "value": 100},
    {"name": "Panjang Badan Lahir", "value": 91},
    {"name": "Berat Badan Saat Ini", "value": 83},
    {"name": "Riwayat Penyakit Infeksi", "value": 74},
    {"name": "Usia", "value": 64},
    {"name": "ASI Eksklusif", "value": 56},
    {"name": "Tinggi Badan Saat Ini", "value": 48},
    {"name": "Status Imunisasi", "value": 40},
]

FORM_FIELDS = (
    "child_name", "gender", "age", "birth_weight", "birth_length",
    "current_weight", "current_height", "exclusive_breastfeeding",
    "immunization_status", "infection_history",
)


def validate_number(value: str, label: str, minimum: float, maximum: float, errors: list[str]) -> None:
    try:
        number = float(value.replace(",", "."))
    except (TypeError, ValueError, AttributeError):
        errors.append(f"{label} harus berupa angka.")
        return
    if not minimum <= number <= maximum:
        errors.append(f"{label} harus berada di antara {minimum:g} dan {maximum:g}.")


def predict_risk(data: dict[str, str]) -> dict:
    """Transparent fallback scoring for the UI until a trained artifact is connected."""
    risk = 0.08
    birth_weight = float(data["birth_weight"].replace(",", "."))
    birth_length = float(data["birth_length"].replace(",", "."))
    age = float(data["age"].replace(",", "."))
    current_weight = float(data["current_weight"].replace(",", "."))
    current_height = float(data["current_height"].replace(",", "."))

    risk += 0.22 if birth_weight < 2500 else 0.10 if birth_weight < 2800 else 0
    if birth_length < 48:
        risk += 0.13
    if data["exclusive_breastfeeding"] == "Tidak":
        risk += 0.12
    if data["immunization_status"] != "Lengkap":
        risk += 0.10
    if data["infection_history"] == "Ada":
        risk += 0.16

    expected_weight = 3.3 + (0.42 * min(age, 24)) + (0.20 * max(age - 24, 0))
    expected_height = 49 + (1.55 * min(age, 12)) + (0.70 * min(max(age - 12, 0), 24)) + (0.45 * max(age - 36, 0))
    if current_weight < expected_weight * 0.80:
        risk += 0.14
    if current_height < expected_height * 0.91:
        risk += 0.18

    probability = max(4, min(96, round(risk * 100)))
    positive = probability >= 50
    return {
        "positive": positive,
        "classification": "STUNTING" if positive else "TIDAK STUNTING",
        "risk_level": "High Risk" if positive else "Low Risk",
        "probability": probability,
        "interpretation": (
            "The child has a high risk of stunting. Parents are advised to consult the nearest healthcare facility for further examination."
            if positive else
            "The child has a low risk of stunting. Continue balanced nutrition and routine growth monitoring at the nearest healthcare facility."
        ),
        "recorded_at": datetime.now().strftime("%d %B %Y, %H:%M"),
        "child_name": data["child_name"],
        "inputs": data,
    }


@app.context_processor
def inject_globals():
    return {"current_year": datetime.now().year}


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
        for field, label in (
            ("gender", "Jenis kelamin"),
            ("exclusive_breastfeeding", "ASI eksklusif"),
            ("immunization_status", "Status imunisasi"),
            ("infection_history", "Riwayat penyakit infeksi"),
        ):
            if not form_data[field]:
                errors.append(f"{label} wajib dipilih.")
        for field, label, minimum, maximum in (
            ("age", "Usia", 0, 60),
            ("birth_weight", "Berat badan lahir", 500, 6000),
            ("birth_length", "Panjang badan lahir", 30, 65),
            ("current_weight", "Berat badan saat ini", 1, 35),
            ("current_height", "Tinggi badan saat ini", 35, 130),
        ):
            validate_number(form_data[field], label, minimum, maximum, errors)
        if not errors:
            session["prediction_result"] = predict_risk(form_data)
            return redirect(url_for("result"))
    return render_template("prediction.html", page_title="Prediksi Risiko — StuntingCare", form_data=form_data, errors=errors)


@app.get("/result")
def result():
    prediction_result = session.get("prediction_result")
    if not prediction_result:
        return redirect(url_for("prediction"))
    return render_template("result.html", page_title="Hasil Prediksi — StuntingCare", result=prediction_result, contributions=FEATURE_CONTRIBUTIONS)


@app.get("/result/download")
def download_result():
    item = session.get("prediction_result")
    if not item:
        return redirect(url_for("prediction"))
    report = (
        "STUNTINGCARE — HASIL PREDIKSI\n================================\n"
        f"Nama Balita : {item['child_name']}\nWaktu       : {item['recorded_at']}\n"
        f"Hasil       : {item['classification']}\nTingkat     : {item['risk_level']}\n"
        f"Probabilitas: {item['probability']}%\n\nInterpretasi\n{item['interpretation']}\n\n"
        "Rekomendasi\n- Routine growth monitoring\n- Improve nutritional intake\n"
        "- Complete immunization\n- Visit Puskesmas regularly\n\n"
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


if __name__ == "__main__":
    app.run(debug=True)
