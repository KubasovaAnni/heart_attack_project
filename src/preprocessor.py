import numpy as np
import pandas as pd

def prepare_features(df_raw: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df_raw.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    binary_cols = [
        "diabetes", "family_history", "smoking", "obesity",
        "alcohol_consumption", "previous_heart_problems", "medication_use"
    ]

    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(stats["binary_fill_value"]).astype(int)
        else:
            df[c] = stats["binary_fill_value"]

    # stress_level -> медиана из stats
    if "stress_level" in df.columns:
        df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")
        df["stress_level"] = df["stress_level"].fillna(stats["stress_level_median"]).astype(int)
    else:
        df["stress_level"] = int(stats["stress_level_median"])

    # physical_activity_days_per_week -> медиана из stats
    if "physical_activity_days_per_week" in df.columns:
        df["physical_activity_days_per_week"] = pd.to_numeric(df["physical_activity_days_per_week"], errors="coerce")
        df["physical_activity_days_per_week"] = (
            df["physical_activity_days_per_week"].fillna(stats["physical_activity_median"]).astype(int)
        )
    else:
        df["physical_activity_days_per_week"] = int(stats["physical_activity_median"])

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.lower()
        df["gender"] = df["gender"].replace({"1.0": "unknown", "0.0": "unknown", "nan": "unknown"})

    # sleep_hours_per_day -> медиана из stats
    if "sleep_hours_per_day" in df.columns:
        df["sleep_hours_per_day"] = pd.to_numeric(df["sleep_hours_per_day"], errors="coerce")
        df["sleep_hours_per_day"] = df["sleep_hours_per_day"].fillna(stats["sleep_hours_median"])
        df["sleep_hours_per_day"] = np.rint(df["sleep_hours_per_day"]).astype(int)
    else:
        df["sleep_hours_per_day"] = int(stats["sleep_hours_median"])

    df["survey_risk_score"] = df[binary_cols].sum(axis=1).astype(int)

    drop_cols = [
        "diabetes", "family_history", "smoking", "obesity",
        "alcohol_consumption", "previous_heart_problems", "medication_use",
        "id", "gender", "troponin", "ck_mb"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


    df["stress_level"] = df["stress_level"].astype(int)
    df["physical_activity_days_per_week"] = df["physical_activity_days_per_week"].astype(int)
    df["survey_risk_score"] = df["survey_risk_score"].astype(int)

    MODEL_COLS = ['stress_level',
    'physical_activity_days_per_week',
    'survey_risk_score',
    'sleep_hours_per_day',
    'diet',
    'age',
    'cholesterol',
    'heart_rate',
    'exercise_hours_per_week',
    'sedentary_hours_per_day',
    'income',
    'bmi',
    'triglycerides',
    'blood_sugar',
    'systolic_blood_pressure',
    'diastolic_blood_pressure',
    ]

    for c in MODEL_COLS:
        if c not in df.columns:
            df[c] = 0

    return df[MODEL_COLS]
