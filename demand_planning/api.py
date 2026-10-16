import os
from datetime import date
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import validate_history
from .features import FEATURES, next_week_features


class Observation(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    week: date
    demand: float = Field(ge=0, allow_inf_nan=False)


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(min_length=52, max_length=50000)


class ExplainRequest(ForecastRequest):
    use_llm: bool = False


def create_app(model_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Demand Planning API", version="0.1.0")
    path = model_path or Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    bundle = None

    def load_model() -> dict:
        nonlocal bundle
        if bundle is None:
            if not path.exists():
                raise HTTPException(503, "Run python -m demand_planning.train first")
            bundle = joblib.load(path)
        return bundle

    def forecast(request: ForecastRequest) -> list[dict]:
        model = load_model()
        try:
            panel = validate_history(pd.DataFrame([row.model_dump() for row in request.history]))
            featured = next_week_features(panel, model["sku_mapping"])
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error
        q50 = np.maximum(0, model["models"]["q50"].predict(featured[FEATURES]))
        q85 = np.maximum(q50, model["models"]["q85"].predict(featured[FEATURES]))
        return [
            {
                "sku": row["sku"], "as_of": str(row["as_of"].date()),
                "target_week": str(row["week"].date()), "horizon_weeks": 1,
                "q50": round(float(q50[index]), 2), "q85": round(float(q85[index]), 2),
            }
            for index, row in featured.iterrows()
        ]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_ready": path.exists()}

    @app.post("/forecast")
    def predict(request: ForecastRequest) -> dict:
        return {"forecasts": forecast(request), "data_mode": "synthetic_demo"}

    @app.post("/explain")
    def explain(request: ExplainRequest) -> dict:
        facts = forecast(request)
        text = "\n".join(
            f"{row['sku']}: неделя {row['target_week']}, медианный прогноз {row['q50']}, квантиль 0.85 — {row['q85']}."
            for row in facts
        )
        if not request.use_llm:
            return {"facts": facts, "explanation": text, "llm_generated": False}
        api_base = os.getenv("LLM_API_BASE")
        api_key = os.getenv("LLM_API_KEY")
        model_name = os.getenv("LLM_MODEL")
        if not all([api_base, api_key, model_name]):
            raise HTTPException(503, "Set LLM_API_BASE, LLM_API_KEY and LLM_MODEL")
        try:
            response = httpx.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model_name, "temperature": 0, "messages": [
                    {"role": "system", "content": "Explain the supplied forecast facts in Russian. Do not calculate or alter forecasts. State that a quantile is not a guarantee. Do not invent causes, business savings or recommendations without inventory data."},
                    {"role": "user", "content": text},
                ]},
                timeout=20,
            )
            response.raise_for_status()
            explanation = response.json()["choices"][0]["message"]["content"]
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("Provider returned no explanation")
        except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError) as error:
            raise HTTPException(502, "LLM provider failed") from error
        return {"facts": facts, "explanation": explanation, "llm_generated": True}

    return app


app = create_app()
