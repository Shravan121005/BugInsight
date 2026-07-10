from fastapi import FastAPI

from .schemas import BugRequest, PredictionResponse
from .predictor import predict_bug

app = FastAPI(
    title="BugInsight API",
    version="1.0.0",
)

@app.get("/")
def home():

    return {
        "message": "BugInsight API is Running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: BugRequest):

    severity, days = predict_bug(request)

    return PredictionResponse(
        severity=severity,
        estimated_fix_time=days
    )