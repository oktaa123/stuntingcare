import os

class Config:
    SECRET_KEY = "stuntingcare_secret_key"

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "model",
        "random_forest.joblib"
    )