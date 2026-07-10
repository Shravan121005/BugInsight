from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

severity_model = joblib.load(MODEL_DIR / "severity_model.pkl")

resolution_model = joblib.load(MODEL_DIR / "resolution_model.pkl")

label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")


def predict_bug(data):

    text = data.summary + " " + data.description

    df = pd.DataFrame([
        {
            "text": text,
            "component_name": data.component_name,
            "product_name": data.product_name,
            "quantity_of_votes": data.quantity_of_votes,
            "quantity_of_comments": data.quantity_of_comments,
        }
    ])

    X = preprocessor.transform(df)

    severity = severity_model.predict(X)

    severity = label_encoder.inverse_transform(severity)[0]

    fix_time = resolution_model.predict(X)[0]

    return severity, round(float(fix_time), 2)