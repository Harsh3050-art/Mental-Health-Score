import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

model = joblib.load("Mental_Score_Model.pkl")


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Mental Health Score Prediction API",
    description="API for predicting a student's mental health score",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# --------------------------------------------------
# Pydantic input model
# --------------------------------------------------

class StudentData(BaseModel):

    age: int = Field(..., ge=10, le=100)

    gender: Literal["Male", "Female"]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    daily_unlocks: int = Field(
        ...,
        ge=0
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )

    stress_level: Literal[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]


# --------------------------------------------------
# Pydantic output model
# --------------------------------------------------

class PredictionResponse(BaseModel):

    predicted_mental_health_score: float


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def greet():

    return {
        "welcome": "Mental Health Score Prediction API"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: StudentData):

    # ----------------------------------------------
    # Feature engineering
    # ----------------------------------------------

    total_productive_hours = (
        data.study_hours +
        data.physical_activity_hours
    )

    sleep_ratio_daily_usage = (
        data.sleep_hours_per_night /
        (data.avg_daily_usage_hours + 1e-6)
    )

    # ----------------------------------------------
    # Create input DataFrame
    # ----------------------------------------------

    input_row = pd.DataFrame([{

        "Age": data.age,

        "Gender": data.gender,

        "Country": data.country,

        "Academic_Level": data.academic_level,

        "Most_Used_Platform": data.most_used_platform,

        "Purpose_Of_Use": data.purpose_of_use,

        "Avg_Daily_Usage_Hours":
            data.avg_daily_usage_hours,

        "Daily_Unlocks":
            data.daily_unlocks,

        "Study_Hours":
            data.study_hours,

        "Physical_Activity_Hours":
            data.physical_activity_hours,

        "Sleep_Hours_Per_Night":
            data.sleep_hours_per_night,

        "Stress_Level":
            data.stress_level,

        "total_productive_hours":
            total_productive_hours,

        "sleep_ratio_daily_usage":
            sleep_ratio_daily_usage

    }])

    # ----------------------------------------------
    # Make prediction
    # ----------------------------------------------

    prediction = model.predict(input_row)[0]

    # ----------------------------------------------
    # Return response
    # ----------------------------------------------

    return PredictionResponse(
        predicted_mental_health_score=round(
            float(prediction),
            2
        )
    )