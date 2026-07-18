import os


class Config:
    # Read the signing key from the environment. The random fallback keeps local
    # development working, but every restart invalidates existing sessions — set
    # SECRET_KEY explicitly in any real deployment.
    SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32).hex()

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Matches the artifact committed to model/ (note: *_model* in the filename).
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        os.path.join(BASE_DIR, "model", "random_forest_model.joblib"),
    )
