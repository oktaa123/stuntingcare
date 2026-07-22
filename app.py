from __future__ import annotations

import os
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from threading import Lock

from flask import Flask, Response, redirect, render_template, request, session, url_for

from config import Config
from utils import scoring

# --- Form specification -------------------------------------------------------
# The form collects 8 model features. Two are continuous z-scores; the other six
# were binary {0, 1} in training, so they are single-choice selects whose option
# VALUES are the integer codes fed to the model.
#
# The matching training notebook confirms these categorical columns are passed
# to the model as 0/1. It does not persist separate encoder objects containing
# the original human-readable meanings, so the existing form encodings remain
# unchanged in this prediction-label fix. See model/README.md.

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

SCREENING_DISCLAIMER = (
    "Hasil skrining ini merupakan alat bantu awal dan bukan diagnosis medis. "
    "Pemeriksaan lanjutan oleh tenaga kesehatan tetap diperlukan."
)

INDONESIAN_MONTHS = (
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
)


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


def format_screening_datetime(value: datetime) -> str:
    """Return a consistent Indonesian date used by the page and PDF report."""
    month = INDONESIAN_MONTHS[value.month - 1]
    return f"{value.day:02d} {month} {value.year}, {value:%H.%M}"


def build_personal_guidance(features: dict[str, float]) -> tuple[list[dict], list[str]]:
    """Build attention factors and one matching recommendation per input condition."""
    reasons: list[dict] = []
    recommendations: list[str] = []

    def add(factor: str, impact: str, description: str, recommendation: str) -> None:
        reasons.append({
            "factor": factor,
            "impact": impact,
            "description": description,
        })
        recommendations.append(recommendation)

    if features["Z Score BB/TB"] <= -3:
        add(
            "Z-Score BB/TB",
            "Sangat Tinggi",
            "Nilai Z-Score BB/TB berada jauh di bawah standar pertumbuhan.",
            "Segera konsultasikan hasil pengukuran BB/TB kepada tenaga kesehatan untuk evaluasi lebih lanjut.",
        )
    elif features["Z Score BB/TB"] <= -2:
        add(
            "Z-Score BB/TB",
            "Tinggi",
            "Nilai Z-Score BB/TB berada di bawah standar pertumbuhan.",
            "Konsultasikan kebutuhan gizi dan pemberian protein hewani sesuai usia kepada tenaga kesehatan.",
        )

    if features["BB/U"] <= -2:
        add(
            "BB/U",
            "Tinggi",
            "Berat badan menurut umur berada di bawah standar pertumbuhan.",
            "Lakukan pemantauan berat badan terjadwal dan evaluasi asupan makan bersama tenaga kesehatan.",
        )

    if features["BB Lahir (gram)"] == 0:
        add(
            "Berat Badan Lahir",
            "Sedang",
            "Terdapat riwayat berat badan lahir rendah (BBLR).",
            "Sampaikan riwayat BBLR saat pemeriksaan agar pertumbuhan balita dapat dipantau lebih intensif.",
        )

    if features["MEROKOK"] == 1:
        add(
            "Paparan Asap Rokok",
            "Sedang",
            "Balita terpapar asap rokok di lingkungan rumah.",
            "Jadikan rumah dan area bermain balita sebagai lingkungan bebas asap rokok.",
        )

    if features["KEPEMILKAN JKN"] == 0:
        add(
            "Kepemilikan JKN",
            "Sedang",
            "Keluarga belum memiliki perlindungan Jaminan Kesehatan Nasional.",
            "Pertimbangkan mengurus kepesertaan JKN untuk mendukung akses pemeriksaan lanjutan.",
        )

    if features["RIWAYAT DATANG POSYANDU"] == 1:
        add(
            "Kunjungan Posyandu",
            "Sedang",
            "Kunjungan pemantauan pertumbuhan ke Posyandu belum rutin.",
            "Susun jadwal kunjungan Posyandu berikutnya agar pertumbuhan dapat dipantau secara berkala.",
        )

    if features["STATUS KELIARGA"] == 0:
        add(
            "Status Keluarga",
            "Sedang",
            "Kondisi keluarga pra-sejahtera dapat membatasi pemenuhan kebutuhan gizi.",
            "Tanyakan program pendampingan atau bantuan gizi yang tersedia di Posyandu maupun Puskesmas.",
        )

    if features["POLA ASUH"] == 1:
        add(
            "Pola Asuh",
            "Sedang",
            "Pola asuh yang dicatat masih memerlukan penguatan.",
            "Konsultasikan pola pemberian makan dan pengasuhan sesuai usia kepada kader atau tenaga kesehatan.",
        )

    return reasons, recommendations


def predict_risk(form_data: dict[str, str], model, now=datetime.now) -> dict:
    """Score one submission with the trained model.

    ``now`` is an injectable clock for deterministic tests.
    """
    features = build_features(form_data)

    if model is None:
        raise RuntimeError("Model Random Forest tidak tersedia untuk proses prediksi.")

    prediction, probability = scoring.predict_class_and_probability(features, model)

    probability_percent = round(probability * 100)
    positive = prediction == scoring.SEVERE_CLASS
    reasons, recommendations = build_personal_guidance(features)
    classification = scoring.classification_for_prediction(prediction)

    if not recommendations:
        recommendations = [
            f"Diskusikan hasil skrining kategori {classification.title()} dengan tenaga kesehatan "
            "untuk menentukan pemantauan yang sesuai."
        ]

    return {
        "positive": positive,
        "classification": classification,
        "probability": probability_percent,
        "source": "model",
        "interpretation": (
            "Berdasarkan data pertumbuhan yang dimasukkan, hasil skrining menunjukkan bahwa "
            f"kondisi balita berada pada kategori {classification.title()}. "
            f"{SCREENING_DISCLAIMER}"
        ),
        "recorded_at": format_screening_datetime(now()),
        "child_name": form_data["child_name"],
        "reasons": reasons,
        "recommendations": recommendations,
    }


def build_result_pdf(item: dict) -> bytes:
    """Build a printable screening report without writing a temporary file."""
    import reportlab
    from reportlab.graphics.shapes import Circle, Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    normal_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    normal_font_path = font_dir / "Vera.ttf"
    bold_font_path = font_dir / "VeraBd.ttf"

    if normal_font_path.exists() and bold_font_path.exists():
        normal_font = "StuntingCareSans"
        bold_font = "StuntingCareSans-Bold"
        registered_fonts = set(pdfmetrics.getRegisteredFontNames())
        if normal_font not in registered_fonts:
            pdfmetrics.registerFont(TTFont(normal_font, str(normal_font_path)))
        if bold_font not in registered_fonts:
            pdfmetrics.registerFont(TTFont(bold_font, str(bold_font_path)))
        pdfmetrics.registerFontFamily(
            normal_font,
            normal=normal_font,
            bold=bold_font,
            italic=normal_font,
            boldItalic=bold_font,
        )

    green = colors.HexColor("#198754")
    green_dark = colors.HexColor("#12653F")
    green_soft = colors.HexColor("#EAF6EF")
    ink = colors.HexColor("#17241D")
    body_color = colors.HexColor("#45584D")
    muted = colors.HexColor("#66766D")
    line = colors.HexColor("#E3EBE6")
    surface = colors.HexColor("#F7FAF8")
    orange = colors.HexColor("#D97706")
    red = colors.HexColor("#C62828")
    yellow = colors.HexColor("#F4C430")

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=24 * mm,
        title="Laporan Hasil Skrining Stunting",
        author="StuntingCare",
    )

    sample_styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "ReportBody",
        parent=sample_styles["BodyText"],
        fontName=normal_font,
        fontSize=9,
        leading=14,
        textColor=body_color,
        spaceAfter=0,
    )
    brand_style = ParagraphStyle(
        "ReportBrand",
        parent=body_style,
        fontSize=8,
        leading=12,
        textColor=muted,
    )
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=body_style,
        fontName=bold_font,
        fontSize=17,
        leading=22,
        alignment=TA_CENTER,
        textColor=ink,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=body_style,
        fontSize=8.5,
        alignment=TA_CENTER,
        textColor=muted,
    )
    section_style = ParagraphStyle(
        "ReportSection",
        parent=body_style,
        fontName=bold_font,
        fontSize=10,
        leading=14,
        textColor=green_dark,
        spaceBefore=2,
        spaceAfter=8,
    )
    label_style = ParagraphStyle(
        "ReportLabel",
        parent=body_style,
        fontName=bold_font,
        fontSize=8,
        textColor=muted,
    )
    center_style = ParagraphStyle(
        "ReportCenter",
        parent=body_style,
        alignment=TA_CENTER,
    )
    badge_style = ParagraphStyle(
        "ReportBadge",
        parent=center_style,
        fontName=bold_font,
        fontSize=7.5,
        leading=10,
    )
    table_header_style = ParagraphStyle(
        "ReportTableHeader",
        parent=center_style,
        fontName=bold_font,
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
    )

    def safe_text(value) -> str:
        return escape(str(value if value is not None else ""))

    logo = Drawing(34, 34)
    logo.add(Circle(17, 17, 16, fillColor=green, strokeColor=None))
    logo.add(String(
        17,
        12.5,
        "SC",
        fontName=bold_font,
        fontSize=9,
        fillColor=colors.white,
        textAnchor="middle",
    ))
    brand = Paragraph(
        f'<font name="{bold_font}" size="14" color="#12653F">StuntingCare</font><br/>'
        '<font size="7.5" color="#66766D">Pendamping tumbuh kembang anak</font>',
        brand_style,
    )
    header = Table([[logo, brand]], colWidths=[15 * mm, 155 * mm])
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    divider = Table([[""]], colWidths=[170 * mm], rowHeights=[1])
    divider.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), green)]))

    story = [
        header,
        Spacer(1, 5 * mm),
        divider,
        Spacer(1, 7 * mm),
        Paragraph("LAPORAN HASIL SKRINING STUNTING", title_style),
        Paragraph("Ringkasan skrining awal pertumbuhan balita", subtitle_style),
        Spacer(1, 8 * mm),
        Paragraph("INFORMASI PEMERIKSAAN", section_style),
    ]

    classification = str(item.get("classification", ""))
    category_color_hex = "#C62828" if classification == "SEVERELY STUNTING" else "#D97706"
    category_soft = colors.HexColor("#FDECEC") if classification == "SEVERELY STUNTING" else colors.HexColor("#FFF3E0")
    data_rows = [
        [Paragraph("Nomor Laporan", label_style), Paragraph(safe_text(item.get("report_number", "-")), body_style)],
        [Paragraph("Tanggal Pemeriksaan", label_style), Paragraph(safe_text(item.get("recorded_at", "-")), body_style)],
        [Paragraph("Nama Balita", label_style), Paragraph(safe_text(item.get("child_name", "-")), body_style)],
        [Paragraph("Kategori Pertumbuhan", label_style), Paragraph(f'<font name="{bold_font}" color="{category_color_hex}">{safe_text(classification)}</font>', body_style)],
        [Paragraph("Probabilitas Prediksi", label_style), Paragraph(f'<font name="{bold_font}">{safe_text(item.get("probability", 0))}%</font>', body_style)],
    ]
    data_table = Table(data_rows, colWidths=[48 * mm, 122 * mm])
    data_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), surface),
        ("BACKGROUND", (1, 3), (1, 3), category_soft),
        ("BOX", (0, 0), (-1, -1), 0.75, line),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([
        data_table,
        Spacer(1, 7 * mm),
        Paragraph("INTERPRETASI", section_style),
        Paragraph(safe_text(item.get("interpretation", "-")), body_style),
        Spacer(1, 7 * mm),
        Paragraph("FAKTOR YANG PERLU DIPERHATIKAN", section_style),
    ])

    reasons = list(item.get("reasons") or [])
    if reasons:
        impact_backgrounds = {
            "Sangat Tinggi": red,
            "Tinggi": orange,
            "Sedang": yellow,
            "Rendah": green,
        }
        reason_rows = [[
            Paragraph("NO.", table_header_style),
            Paragraph("KONDISI YANG DITEMUKAN", table_header_style),
            Paragraph("PERHATIAN", table_header_style),
        ]]
        reason_styles = [
            ("BOX", (0, 0), (-1, -1), 0.75, line),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 1), (0, -1), green_soft),
            ("BACKGROUND", (0, 0), (-1, 0), green_dark),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]
        for index, reason in enumerate(reasons, start=1):
            impact = str(reason.get("impact", ""))
            impact_background = impact_backgrounds.get(impact, muted)
            impact_text_color = "#212529" if impact == "Sedang" else "#FFFFFF"
            reason_rows.append([
                Paragraph(f"{index:02d}", center_style),
                Paragraph(
                    f'<font name="{bold_font}" color="#17241D">{safe_text(reason.get("factor", "-"))}</font><br/>'
                    f'<font size="8" color="#66766D">{safe_text(reason.get("description", ""))}</font>',
                    body_style,
                ),
                Paragraph(
                    f'<font color="{impact_text_color}">{safe_text(impact)}</font>',
                    badge_style,
                ),
            ])
            reason_styles.append(("BACKGROUND", (2, index), (2, index), impact_background))

        reason_table = Table(reason_rows, colWidths=[13 * mm, 122 * mm, 35 * mm], repeatRows=1)
        reason_table.setStyle(TableStyle(reason_styles))
        story.append(reason_table)
    else:
        story.append(Paragraph(
            "Tidak ada faktor khusus yang memenuhi kondisi perhatian pada data yang dimasukkan.",
            body_style,
        ))

    story.extend([
        Spacer(1, 7 * mm),
        Paragraph("REKOMENDASI", section_style),
    ])
    recommendation_rows = []
    for index, recommendation in enumerate(item.get("recommendations") or [], start=1):
        recommendation_rows.append([
            Paragraph(str(index), center_style),
            Paragraph(safe_text(recommendation), body_style),
        ])
    if recommendation_rows:
        recommendation_table = Table(recommendation_rows, colWidths=[13 * mm, 157 * mm])
        recommendation_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, line),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (0, -1), green_soft),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(recommendation_table)

    note = Paragraph(
        f'<font name="{bold_font}" color="#12653F">CATATAN</font><br/>' +
        safe_text(SCREENING_DISCLAIMER),
        body_style,
    )
    note_table = Table([[note]], colWidths=[170 * mm])
    note_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), green_soft),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CFE6D7")),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([Spacer(1, 7 * mm), note_table])

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setTitle("Laporan Hasil Skrining Stunting")
        canvas.setAuthor("StuntingCare")
        if doc.page > 1:
            canvas.setFont(bold_font, 8)
            canvas.setFillColor(green_dark)
            canvas.drawString(doc.leftMargin, A4[1] - 12 * mm, "StuntingCare")
            canvas.setFont(normal_font, 7)
            canvas.setFillColor(muted)
            canvas.drawRightString(
                A4[0] - doc.rightMargin,
                A4[1] - 12 * mm,
                safe_text(item.get("report_number", "Laporan Skrining")),
            )
            canvas.setStrokeColor(line)
            canvas.line(doc.leftMargin, A4[1] - 15 * mm, A4[0] - doc.rightMargin, A4[1] - 15 * mm)
        canvas.setStrokeColor(line)
        canvas.setLineWidth(0.6)
        canvas.line(doc.leftMargin, 17 * mm, A4[0] - doc.rightMargin, 17 * mm)
        canvas.setFont(normal_font, 7)
        canvas.setFillColor(muted)
        canvas.drawString(doc.leftMargin, 11 * mm, "Universitas Muhammadiyah Semarang")
        canvas.drawCentredString(A4[0] / 2, 11 * mm, "StuntingCare")
        canvas.drawRightString(A4[0] - doc.rightMargin, 11 * mm, "2026")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()


def create_app(config_object: type = Config) -> Flask:
    """Application factory so tests can build an isolated app with a test config."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Load the trained artifact once while the Flask application starts. Route
    # handlers reuse this in-memory instance for every screening.
    if "MODEL" not in app.config:
        app.config["MODEL"] = scoring.load_model(app.config["MODEL_PATH"])

    report_counters: dict[str, int] = {}
    report_counter_lock = Lock()

    def clock():
        return app.config.get("CLOCK", datetime.now)()

    def next_report_number(screened_at: datetime) -> str:
        date_key = screened_at.strftime("%Y%m%d")
        with report_counter_lock:
            report_counters[date_key] = report_counters.get(date_key, 0) + 1
            sequence = report_counters[date_key]
        return f"SCR-{date_key}-{sequence:04d}"

    @app.context_processor
    def inject_globals():
        return {"current_year": clock().year}

    @app.after_request
    def prevent_stale_screening_cache(response):
        if request.endpoint in {"prediction", "result", "download_result"}:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def home():
        return render_template("home.html", page_title="StuntingCare — Deteksi Dini, Tumbuh Optimal")

    @app.route("/prediction", methods=["GET", "POST"])
    def prediction():
        # Starting a screening, including via "Skrining Lagi", invalidates the
        # previous result and always presents an empty form.
        session.pop("prediction_result", None)
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
                screened_at = clock()
                prediction_result = predict_risk(
                    form_data,
                    app.config["MODEL"],
                    now=lambda: screened_at,
                )
                prediction_result["report_number"] = next_report_number(screened_at)
                session["prediction_result"] = prediction_result
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
        )

    @app.get("/result/download")
    def download_result():
        item = session.get("prediction_result")
        if not item:
            return redirect(url_for("prediction"))
        report = build_result_pdf(item)
        filename = f"{item.get('report_number', 'StuntingCare')}-hasil-skrining.pdf"
        return Response(
            report,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

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
