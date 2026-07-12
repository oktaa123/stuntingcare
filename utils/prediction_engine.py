from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FORM_OPTIONS = {
    "bbu_status": [
        {"value": "sangat_kurang", "label": "Sangat Kurang"},
        {"value": "kurang", "label": "Kurang"},
        {"value": "normal", "label": "Normal"},
        {"value": "lebih", "label": "Lebih"},
    ],
    "has_jkn": [
        {"value": "ya", "label": "Ya"},
        {"value": "tidak", "label": "Tidak"},
    ],
    "posyandu_visit_history": [
        {"value": "rutin", "label": "Rutin"},
        {"value": "kadang", "label": "Kadang-kadang"},
        {"value": "tidak_rutin", "label": "Tidak Rutin"},
    ],
    "family_status": [
        {"value": "pra_sejahtera", "label": "Pra-Sejahtera"},
        {"value": "rentan", "label": "Rentan"},
        {"value": "sejahtera", "label": "Sejahtera"},
    ],
    "kb_status": [
        {"value": "ya", "label": "Ya"},
        {"value": "tidak", "label": "Tidak"},
    ],
}

FIELD_META = {
    "birth_weight": {
        "label": "BB Lahir",
        "unit": "gram",
        "description": "Berat lahir rendah berkaitan dengan gangguan tumbuh kembang sejak awal kehidupan.",
        "min": 500,
        "max": 6000,
    },
    "bbu_status": {
        "label": "BB/U",
        "description": "Status berat badan menurut umur memberi gambaran kecukupan gizi saat ini.",
    },
    "z_score_weight_height": {
        "label": "Z Score BB/TB",
        "description": "Nilai z score yang rendah menunjukkan keseimbangan pertumbuhan yang perlu dipantau.",
        "min": -6,
        "max": 6,
    },
    "has_jkn": {
        "label": "Kepemilikan JKN",
        "description": "Jaminan kesehatan membantu akses keluarga ke pemeriksaan dan tindak lanjut medis.",
    },
    "age_days": {
        "label": "Umur",
        "unit": "hari",
        "description": "Umur diperlukan untuk membaca indikator pertumbuhan secara tepat sesuai tahap perkembangan.",
        "min": 0,
        "max": 1825,
    },
    "posyandu_visit_history": {
        "label": "Riwayat Datang Posyandu",
        "description": "Kunjungan rutin membuat pemantauan pertumbuhan dan edukasi keluarga lebih konsisten.",
    },
    "family_status": {
        "label": "Status Keluarga",
        "description": "Kondisi keluarga berpengaruh pada kualitas asupan, sanitasi, dan kemampuan akses layanan.",
    },
    "kb_status": {
        "label": "KB",
        "description": "Keluarga berencana mendukung pengasuhan dan pengaturan jarak kelahiran yang lebih baik.",
    },
}

DEFAULT_FORM_DATA = {field: "" for field in FIELD_META}

DISPLAY_LABELS = {
    field: {option["value"]: option["label"] for option in options}
    for field, options in FORM_OPTIONS.items()
}

MODEL_COLUMNS = [
    "bb_lahir",
    "bb_u",
    "z_score_bb_tb",
    "kepemilikan_jkn",
    "umur_hari",
    "riwayat_datang_posyandu",
    "status_keluarga",
    "kb",
]

FEATURE_ALIASES = {
    "birth_weight": ["birth_weight", "bb_lahir", "berat_lahir"],
    "bbu_status": ["bbu_status", "bb_u", "status_bbu"],
    "z_score_weight_height": ["z_score_weight_height", "z_score_bb_tb", "zscore_bb_tb"],
    "has_jkn": ["has_jkn", "kepemilikan_jkn", "jkn"],
    "age_days": ["age_days", "umur_hari", "umur"],
    "posyandu_visit_history": [
        "posyandu_visit_history",
        "riwayat_datang_posyandu",
        "riwayat_posyandu",
    ],
    "family_status": ["family_status", "status_keluarga"],
    "kb_status": ["kb_status", "kb"],
}

NUMERIC_ENCODINGS = {
    "bbu_status": {
        "sangat_kurang": 0,
        "kurang": 1,
        "normal": 2,
        "lebih": 3,
    },
    "has_jkn": {"tidak": 0, "ya": 1},
    "posyandu_visit_history": {
        "tidak_rutin": 0,
        "kadang": 1,
        "rutin": 2,
    },
    "family_status": {
        "pra_sejahtera": 0,
        "rentan": 1,
        "sejahtera": 2,
    },
    "kb_status": {"tidak": 0, "ya": 1},
}

HIGH_RISK_RECOMMENDATIONS = [
    "Datang ke Posyandu setiap bulan.",
    "Konsultasi ke Puskesmas untuk evaluasi lanjutan.",
    "Perbaiki pola makan dan pantau asupan gizi harian.",
    "Pantau pertumbuhan berat badan dan tinggi badan secara berkala.",
    "Lakukan pengukuran ulang untuk memastikan kondisi anak.",
]

LOW_RISK_RECOMMENDATIONS = [
    "Pertahankan pola makan yang seimbang dan sesuai usia.",
    "Tetap kontrol rutin ke Posyandu atau fasilitas kesehatan.",
    "Pantau tinggi badan dan berat badan secara berkala.",
]


def get_default_form_data() -> dict[str, str]:
    return DEFAULT_FORM_DATA.copy()


def collect_form_data(source: Any) -> dict[str, str]:
    return {field: str(source.get(field, "")).strip() for field in FIELD_META}


def validate_and_clean_form_data(form_data: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    cleaned["birth_weight"] = _parse_number(
        form_data.get("birth_weight"),
        "BB Lahir",
        errors,
        minimum=FIELD_META["birth_weight"]["min"],
        maximum=FIELD_META["birth_weight"]["max"],
        integer_only=True,
    )
    cleaned["z_score_weight_height"] = _parse_number(
        form_data.get("z_score_weight_height"),
        "Z Score BB/TB",
        errors,
        minimum=FIELD_META["z_score_weight_height"]["min"],
        maximum=FIELD_META["z_score_weight_height"]["max"],
    )
    cleaned["age_days"] = _parse_number(
        form_data.get("age_days"),
        "Umur",
        errors,
        minimum=FIELD_META["age_days"]["min"],
        maximum=FIELD_META["age_days"]["max"],
        integer_only=True,
    )

    for field in ("bbu_status", "has_jkn", "posyandu_visit_history", "family_status", "kb_status"):
        value = form_data.get(field, "")
        allowed_values = DISPLAY_LABELS[field]
        if value not in allowed_values:
            errors.append(f"{FIELD_META[field]['label']} wajib dipilih.")
        else:
            cleaned[field] = value

    return cleaned, errors


def build_prediction_result(cleaned_data: dict[str, Any], model_path: str) -> dict[str, Any]:
    probability, source = _predict_probability(cleaned_data, model_path)
    probability = max(0.01, min(round(float(probability), 4), 0.99))
    is_high_risk = probability >= 0.5
    probability_percent = int(round(probability * 100))

    return {
        "classification_text": "Terindikasi Stunting" if is_high_risk else "Tidak Terindikasi Stunting",
        "status_text": "Risiko Tinggi" if is_high_risk else "Risiko Rendah",
        "status_variant": "high" if is_high_risk else "low",
        "status_icon": "bi-exclamation-triangle-fill" if is_high_risk else "bi-shield-check",
        "status_summary": (
            "Model mengarah pada kelas stunting sehingga perlu tindak lanjut pemantauan dan intervensi lebih dekat."
            if is_high_risk
            else "Model belum mengarah pada kelas stunting, namun pemantauan rutin tetap diperlukan."
        ),
        "probability": probability,
        "probability_percent": probability_percent,
        "probability_bar_class": "danger" if is_high_risk else "success",
        "inputs": _build_input_summary(cleaned_data),
        "factor_explanations": _build_factor_explanations(cleaned_data),
        "recommendations": HIGH_RISK_RECOMMENDATIONS if is_high_risk else LOW_RISK_RECOMMENDATIONS,
        "source": source,
    }


def _predict_probability(cleaned_data: dict[str, Any], model_path: str) -> tuple[float, str]:
    model = _load_model(model_path)
    if model is not None:
        try:
            return _predict_with_model(model, cleaned_data), "random_forest"
        except Exception:
            pass

    return _fallback_probability(cleaned_data), "fallback"


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None
    return joblib.load(path)


def _predict_with_model(model: Any, cleaned_data: dict[str, Any]) -> float:
    for candidate in _build_model_candidates(model, cleaned_data):
        probability = _extract_probability(model, candidate)
        if probability is not None:
            return probability

    raise RuntimeError("Model tidak kompatibel dengan format input saat ini.")


def _build_model_candidates(model: Any, cleaned_data: dict[str, Any]) -> list[Any]:
    label_features = {
        "birth_weight": cleaned_data["birth_weight"],
        "bbu_status": DISPLAY_LABELS["bbu_status"][cleaned_data["bbu_status"]],
        "z_score_weight_height": cleaned_data["z_score_weight_height"],
        "has_jkn": DISPLAY_LABELS["has_jkn"][cleaned_data["has_jkn"]],
        "age_days": cleaned_data["age_days"],
        "posyandu_visit_history": DISPLAY_LABELS["posyandu_visit_history"][cleaned_data["posyandu_visit_history"]],
        "family_status": DISPLAY_LABELS["family_status"][cleaned_data["family_status"]],
        "kb_status": DISPLAY_LABELS["kb_status"][cleaned_data["kb_status"]],
    }

    code_features = {
        "birth_weight": cleaned_data["birth_weight"],
        "bbu_status": cleaned_data["bbu_status"],
        "z_score_weight_height": cleaned_data["z_score_weight_height"],
        "has_jkn": cleaned_data["has_jkn"],
        "age_days": cleaned_data["age_days"],
        "posyandu_visit_history": cleaned_data["posyandu_visit_history"],
        "family_status": cleaned_data["family_status"],
        "kb_status": cleaned_data["kb_status"],
    }

    numeric_features = {
        "birth_weight": cleaned_data["birth_weight"],
        "bbu_status": NUMERIC_ENCODINGS["bbu_status"][cleaned_data["bbu_status"]],
        "z_score_weight_height": cleaned_data["z_score_weight_height"],
        "has_jkn": NUMERIC_ENCODINGS["has_jkn"][cleaned_data["has_jkn"]],
        "age_days": cleaned_data["age_days"],
        "posyandu_visit_history": NUMERIC_ENCODINGS["posyandu_visit_history"][cleaned_data["posyandu_visit_history"]],
        "family_status": NUMERIC_ENCODINGS["family_status"][cleaned_data["family_status"]],
        "kb_status": NUMERIC_ENCODINGS["kb_status"][cleaned_data["kb_status"]],
    }

    feature_names = list(getattr(model, "feature_names_in_", MODEL_COLUMNS))

    candidates = [
        _build_dataframe(feature_names, label_features),
        _build_dataframe(feature_names, code_features),
        _build_dataframe(feature_names, numeric_features),
        pd.DataFrame([[numeric_features[key] for key in numeric_features]], columns=list(numeric_features)),
        np.array([[numeric_features["birth_weight"], numeric_features["bbu_status"], numeric_features["z_score_weight_height"], numeric_features["has_jkn"], numeric_features["age_days"], numeric_features["posyandu_visit_history"], numeric_features["family_status"], numeric_features["kb_status"]]]),
    ]

    return candidates


def _build_dataframe(feature_names: list[str], source: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for feature_name in feature_names:
        canonical_name = _resolve_canonical_name(feature_name)
        if canonical_name in source:
            row[feature_name] = source[canonical_name]
    return pd.DataFrame([row])


def _resolve_canonical_name(feature_name: str) -> str:
    for canonical_name, aliases in FEATURE_ALIASES.items():
        if feature_name in aliases:
            return canonical_name
    return feature_name


def _extract_probability(model: Any, candidate: Any) -> float | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(candidate))[0]
        if probabilities.ndim == 0:
            return float(probabilities)
        if len(probabilities) == 1:
            return float(probabilities[0])
        high_risk_index = _get_high_risk_index(getattr(model, "classes_", []), len(probabilities))
        return float(probabilities[high_risk_index])

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(candidate)).flatten()[0]
        return 1 / (1 + math.exp(-float(decision)))

    if hasattr(model, "predict"):
        predicted_label = model.predict(candidate)[0]
        return 0.8 if _is_high_risk_label(predicted_label) else 0.2

    return None


def _get_high_risk_index(classes: Any, total: int) -> int:
    if classes is None or len(classes) != total:
        return total - 1

    for index, class_name in enumerate(classes):
        if _is_high_risk_label(class_name):
            return index

    return total - 1


def _is_high_risk_label(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return float(value) >= 1

    normalized = str(value).strip().lower()
    return normalized in {
        "1",
        "true",
        "high",
        "high risk",
        "risiko tinggi",
        "tinggi",
        "stunting",
        "yes",
    }


def _fallback_probability(cleaned_data: dict[str, Any]) -> float:
    score = -2.2

    if cleaned_data["birth_weight"] < 2500:
        score += 2.4
    elif cleaned_data["birth_weight"] < 3000:
        score += 1.0

    if cleaned_data["bbu_status"] == "sangat_kurang":
        score += 2.5
    elif cleaned_data["bbu_status"] == "kurang":
        score += 1.6
    elif cleaned_data["bbu_status"] == "normal":
        score += 0.3

    if cleaned_data["z_score_weight_height"] < -3:
        score += 2.4
    elif cleaned_data["z_score_weight_height"] < -2:
        score += 1.7
    elif cleaned_data["z_score_weight_height"] < -1:
        score += 0.9

    if cleaned_data["has_jkn"] == "tidak":
        score += 0.7

    if cleaned_data["age_days"] > 730:
        score += 0.5
    elif cleaned_data["age_days"] < 180:
        score += 0.2

    if cleaned_data["posyandu_visit_history"] == "tidak_rutin":
        score += 1.5
    elif cleaned_data["posyandu_visit_history"] == "kadang":
        score += 0.9

    if cleaned_data["family_status"] == "pra_sejahtera":
        score += 1.5
    elif cleaned_data["family_status"] == "rentan":
        score += 0.9

    if cleaned_data["kb_status"] == "tidak":
        score += 0.6

    return 1 / (1 + math.exp(-score))


def _build_input_summary(cleaned_data: dict[str, Any]) -> list[dict[str, str]]:
    summary = []
    for field_name, meta in FIELD_META.items():
        summary.append(
            {
                "label": meta["label"],
                "value": _format_field_value(field_name, cleaned_data[field_name]),
                "description": meta["description"],
            }
        )
    return summary


def _build_factor_explanations(cleaned_data: dict[str, Any]) -> list[dict[str, str]]:
    explanations = []
    for field_name, meta in FIELD_META.items():
        impact = _calculate_factor_impact(field_name, cleaned_data[field_name])
        explanations.append(
            {
                "label": meta["label"],
                "value": _format_field_value(field_name, cleaned_data[field_name]),
                "impact_label": impact["label"],
                "impact_class": impact["class"],
                "description": impact["description"],
            }
        )
    return explanations


def _calculate_factor_impact(field_name: str, value: Any) -> dict[str, str]:
    if field_name == "birth_weight":
        if value < 2500:
            return _impact("Meningkatkan risiko", "high", "Berat lahir rendah merupakan faktor penting yang perlu diwaspadai.")
        if value < 3000:
            return _impact("Perlu dipantau", "medium", "Berat lahir mendekati batas bawah sehingga pemantauan lanjutan tetap disarankan.")
        return _impact("Relatif aman", "low", "Berat lahir berada pada rentang yang lebih baik untuk pertumbuhan awal.")

    if field_name == "bbu_status":
        if value == "sangat_kurang":
            return _impact("Meningkatkan risiko", "high", "Status BB/U sangat kurang menunjukkan kondisi gizi yang perlu ditangani segera.")
        if value == "kurang":
            return _impact("Perlu dipantau", "medium", "Status BB/U kurang menunjukkan kecukupan gizi belum optimal.")
        return _impact("Relatif aman", "low", "Status BB/U lebih baik untuk mendukung pertumbuhan.")

    if field_name == "z_score_weight_height":
        if value < -3:
            return _impact("Meningkatkan risiko", "high", "Z score sangat rendah menunjukkan masalah pertumbuhan yang kuat.")
        if value < -2:
            return _impact("Perlu dipantau", "medium", "Z score rendah memberi sinyal perlunya pemantauan lebih dekat.")
        return _impact("Relatif aman", "low", "Z score berada pada rentang yang lebih baik.")

    if field_name == "has_jkn":
        if value == "tidak":
            return _impact("Akses terbatas", "medium", "Tanpa JKN, keluarga mungkin lebih sulit mengakses layanan kesehatan lanjutan.")
        return _impact("Mendukung proteksi", "low", "Kepemilikan JKN membantu akses pemeriksaan dan rujukan.")

    if field_name == "age_days":
        if value <= 730:
            return _impact("Periode penting", "medium", "Usia dini merupakan fase emas pertumbuhan sehingga intervensi cepat sangat bermakna.")
        return _impact("Konteks analisis", "low", "Usia anak membantu menentukan interpretasi hasil pertumbuhan yang lebih tepat.")

    if field_name == "posyandu_visit_history":
        if value == "tidak_rutin":
            return _impact("Meningkatkan risiko", "high", "Kunjungan yang tidak rutin membuat deteksi dini dan edukasi keluarga lebih sulit.")
        if value == "kadang":
            return _impact("Perlu dipantau", "medium", "Kunjungan yang belum konsisten perlu ditingkatkan agar pemantauan lebih teratur.")
        return _impact("Mendukung proteksi", "low", "Kunjungan rutin membantu pemantauan pertumbuhan dan tindak lanjut.")

    if field_name == "family_status":
        if value == "pra_sejahtera":
            return _impact("Meningkatkan risiko", "high", "Kondisi pra-sejahtera bisa memengaruhi kualitas asupan, sanitasi, dan layanan kesehatan.")
        if value == "rentan":
            return _impact("Perlu dipantau", "medium", "Status keluarga rentan tetap memerlukan dukungan pemantauan dan edukasi.")
        return _impact("Mendukung proteksi", "low", "Kondisi keluarga lebih stabil mendukung pemenuhan kebutuhan dasar anak.")

    if field_name == "kb_status":
        if value == "tidak":
            return _impact("Perlu dipantau", "medium", "Status KB dapat berkaitan dengan kesiapan pengasuhan dan jarak kelahiran.")
        return _impact("Mendukung proteksi", "low", "Pengaturan keluarga dapat mendukung perhatian yang lebih optimal pada anak.")

    return _impact("Konteks analisis", "low", "Variabel ini tetap digunakan untuk melengkapi pembacaan kondisi anak.")


def _impact(label: str, css_class: str, description: str) -> dict[str, str]:
    return {"label": label, "class": css_class, "description": description}


def _format_field_value(field_name: str, value: Any) -> str:
    if field_name in DISPLAY_LABELS:
        return DISPLAY_LABELS[field_name].get(str(value), str(value))

    if field_name == "birth_weight":
        return f"{int(value)} gram"

    if field_name == "age_days":
        return f"{int(value)} hari"

    if field_name == "z_score_weight_height":
        return f"{float(value):.2f}"

    return str(value)


def _parse_number(
    raw_value: str | None,
    label: str,
    errors: list[str],
    minimum: float,
    maximum: float,
    integer_only: bool = False,
) -> float | int:
    if raw_value is None or str(raw_value).strip() == "":
        errors.append(f"{label} wajib diisi.")
        return 0

    try:
        parsed_value = float(str(raw_value).strip())
    except ValueError:
        errors.append(f"{label} harus berupa angka yang valid.")
        return 0

    if parsed_value < minimum or parsed_value > maximum:
        errors.append(f"{label} harus berada pada rentang {minimum:g} sampai {maximum:g}.")

    if integer_only:
        if not parsed_value.is_integer():
            errors.append(f"{label} harus berupa bilangan bulat.")
        return int(parsed_value)

    return parsed_value
