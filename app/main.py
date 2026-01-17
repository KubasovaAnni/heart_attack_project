from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Optional

from src.preprocessor import prepare_features

app = FastAPI(title="Heart Attack Risk API")

# Пути (корень проекта = папка, где лежат app/, src/, models/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
STATS_PATH = PROJECT_ROOT / "models" / "preprocess_stats.json"

# Загружаем модель и stats при старте сервера
model = joblib.load(MODEL_PATH)
with open(STATS_PATH, "r", encoding="utf-8") as f:
    stats = json.load(f)


class PatientRaw(BaseModel):
    # --- ВАЖНО: порядок соответствует входу модели ---

    stress_level: Optional[float] = None
    physical_activity_days_per_week: Optional[float] = None
    sleep_hours_per_day: Optional[float] = None
    diet: int

    age: float
    cholesterol: float
    heart_rate: float
    exercise_hours_per_week: float
    sedentary_hours_per_day: float
    income: float
    bmi: float
    triglycerides: float
    blood_sugar: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float

    # survey_risk_score НЕ вводится пользователем,
    # он будет рассчитан в preprocess
    # --- анкетные (сырые), для расчёта survey_risk_score ---
    diabetes: Optional[float] = None
    family_history: Optional[float] = None
    smoking: Optional[float] = None
    obesity: Optional[float] = None
    alcohol_consumption: Optional[float] = None
    previous_heart_problems: Optional[float] = None
    medication_use: Optional[float] = None

@app.get("/")
def root():
    return {"message": "API работает. Открой /docs"}


@app.post("/predict")
def predict(patient: PatientRaw):
    try:
        df_raw = pd.DataFrame([patient.model_dump()])
        X = prepare_features(df_raw, stats)
        proba = float(model.predict_proba(X)[0][1])
        pred = int(proba >= 0.5)
        return {"prediction": pred, "probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
