from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"

NUMERIC_FIELDS = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Study_Hours",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
]


class PredictionRequest(BaseModel):
    age: int = Field(ge=10, le=100)
    gender: str = Field(min_length=1, max_length=40)
    country: str = Field(min_length=1, max_length=80)
    academic_level: str = Field(min_length=1, max_length=40)
    most_used_platform: str = Field(min_length=1, max_length=40)
    purpose_of_use: str = Field(min_length=1, max_length=40)
    avg_daily_usage_hours: float = Field(ge=0, le=24)
    daily_unlocks: int = Field(ge=0, le=2000)
    study_hours: float = Field(ge=0, le=24)
    physical_activity_hours: float = Field(ge=0, le=24)
    sleep_hours_per_night: float = Field(ge=0, le=24)
    stress_level: str = Field(min_length=1, max_length=40)


class ModelService:
    def __init__(self) -> None:
        self.model = joblib.load(MODELS_DIR / "model.pkl")
        self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        self.feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
        self.model_features = joblib.load(MODELS_DIR / "model_features.pkl")
        self.dataset = pd.read_csv(DATA_PATH)
        self.top_countries = set(
            self.dataset["Country"].value_counts().head(11).index.tolist()
        )

    def options(self) -> dict[str, list[str]]:
        return {
            "gender": self.values("Gender"),
            "country": self.values("Country"),
            "academic_level": self.values("Academic_Level"),
            "most_used_platform": self.values("Most_Used_Platform"),
            "purpose_of_use": self.values("Purpose_Of_Use"),
            "stress_level": ["Low", "Medium", "High", "Very High"],
        }

    def values(self, column: str) -> list[str]:
        return sorted(self.dataset[column].dropna().astype(str).unique().tolist())

    def predict(self, request: PredictionRequest) -> float:
        payload: dict[str, Any] = {
            "Age": request.age,
            "Gender": request.gender,
            "Country": request.country,
            "Academic_Level": request.academic_level,
            "Most_Used_Platform": request.most_used_platform,
            "Purpose_Of_Use": request.purpose_of_use,
            "Avg_Daily_Usage_Hours": request.avg_daily_usage_hours,
            "Daily_Unlocks": request.daily_unlocks,
            "Study_Hours": request.study_hours,
            "Physical_Activity_Hours": request.physical_activity_hours,
            "Sleep_Hours_Per_Night": request.sleep_hours_per_night,
            "Stress_Level": request.stress_level,
        }
        row = pd.DataFrame([payload])
        row["Grouped_Country"] = row["Country"].where(
            row["Country"].isin(self.top_countries), "Other"
        )
        transformed = self.preprocessor.transform(row)
        transformed = pd.DataFrame(transformed, columns=self.feature_names)
        transformed = transformed.reindex(columns=self.model_features, fill_value=0)
        return float(self.model.predict(transformed)[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = ModelService()
    yield


app = FastAPI(
    title="Mental Health Score Predictor",
    description="Predict a mental health score from lifestyle and social media habits.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict[str, str]:
    service = getattr(request.app.state, "model_service", None)
    return {"status": "ok" if service else "degraded"}


@app.get("/api/options")
def options(request: Request) -> dict[str, list[str]]:
    return request.app.state.model_service.options()


@app.post("/api/predict")
def predict(payload: PredictionRequest, request: Request) -> dict[str, float]:
    try:
        score = request.app.state.model_service.predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"mental_health_score": round(score, 2)}
