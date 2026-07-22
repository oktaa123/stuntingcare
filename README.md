# StuntingCare

Aplikasi web **skrining dini risiko stunting balita**. Pengguna mengisi data
antropometri (Z-score) dan kondisi sosial anak, lalu sebuah model **Random
Forest** memperkirakan probabilitas risiko stunting beserta rekomendasi tindak
lanjut. Hasil skrining bersifat pendamping, **bukan diagnosis medis**.

Dibangun dengan **Flask** (server-rendered Jinja) + **scikit-learn**.

## Fitur

- Form skrining 8 fitur → prediksi model (`STUNTING` / `SEVERELY STUNTING` + probabilitas).
- Halaman hasil: interpretasi, rekomendasi, dan **feature importance asli** dari model.
- Unduh hasil sebagai berkas teks.
- Prediksi hanya dijalankan dengan artefak Random Forest; aplikasi tidak membuat kategori pengganti secara hardcode.

## Struktur

| Path | Isi |
|------|-----|
| `app.py` | Factory `create_app`, route, validasi form, wiring model |
| `config.py` | Konfigurasi (secret dari env, path model) |
| `utils/scoring.py` | Load model, prediksi kelas, probabilitas kelas hasil, feature importance |
| `templates/` | Halaman Jinja (`layout` = base) |
| `model/random_forest_model.joblib` | Artefak model terlatih — lihat `model/README.md` |
| `tests/` | Suite pytest untuk route, scoring, mapping label, dan konsistensi model |

## Menjalankan (lokal)

Butuh **Python 3.12–3.14**.

```bash
# 1. Buat virtualenv + install dependency
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Jalankan server (port 5000 di macOS sering dipakai AirPlay → pakai 5001)
.venv/bin/python -m flask --app app run --debug --port 5001
```

Buka http://127.0.0.1:5001

Variabel lingkungan opsional:

- `SECRET_KEY` — kunci sesi (default: acak per-restart untuk dev).
- `FLASK_DEBUG=1` — aktifkan debug saat menjalankan `python app.py`.
- `MODEL_PATH` — override lokasi berkas model.

## Menjalankan test

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Catatan model

- Model dilatih dengan **scikit-learn 1.6.1**, dijalankan pada **1.9.0** (unpickle
  dengan warning). Untuk produksi: latih ulang / samakan versi + Python 3.12.
- ⚠️ Peta encoding 6 fitur kategorikal masih **asumsi** (notebook training belum
  ada). Dua fitur z-score (~85% bobot model) tidak terpengaruh. Detail +
  cara mengoreksi: `model/README.md`.

## Deploy

`gunicorn` sudah tersedia sebagai dependency:

```bash
.venv/bin/gunicorn "app:app" --bind 0.0.0.0:8000
```

Untuk produksi, set `SECRET_KEY` eksplisit dan jangan aktifkan debug.
