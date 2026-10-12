"""Pydantic request/response schemas. Transaction fields are exactly the model features; unknown or missing
fields are rejected (never silently reordered, defaulted or invented)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.data.schema import FEATURES


def _field(name: str):
    if name == "Amount":
        return (float, Field(..., ge=0, le=1e7, allow_inf_nan=False, description="Transaction amount (>= 0)"))
    return (float, Field(..., ge=-1000, le=1000, allow_inf_nan=False,
                         description="Anonymised PCA component (no business meaning)"))


Transaction = create_model("Transaction", __config__=ConfigDict(extra="forbid"), **{f: _field(f) for f in FEATURES})
MonitorEvent = create_model(
    "MonitorEvent", __base__=Transaction,
    label=(Literal[0, 1] | None, Field(None, description="Ground truth if (and only if) it is actually known")))


class Prediction(BaseModel):
    fraud_probability: float = Field(description="XGBoost P(fraud). Supervised, label-trained.")
    anomaly_score: float = Field(description="Isolation Forest percentile rank in [0,1] vs training data. "
                                             "Unsupervised; NOT a probability.")
    decision_score: float = Field(description="Score compared with the threshold (xgboost prob or ensemble)")
    risk_category: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW"]
    decision_model: str
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=500)  # type: ignore[valid-type]


class EventsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[MonitorEvent] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]
    source: Literal["external", "simulated"] = "external"


class EvaluateRequest(BaseModel):
    window: int | None = Field(None, ge=10, le=50000)
    source: Literal["live", "external", "simulated"] | None = None
    auto_trigger: bool | None = Field(None, description="Override SENTINEL_AUTO_RETRAIN_ON_DRIFT for this call")


class SimulateRequest(BaseModel):
    n: int = Field(500, ge=50, le=5000)
    features: list[str] | None = None
    magnitude: float = Field(2.0, ge=0, le=10, description="Shift in standard deviations")
    amount_factor: float = Field(3.0, ge=0.1, le=100)
    seed: int = 42
    auto_trigger: bool | None = None


class RetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str | None = Field(None, max_length=300)
    overrides: dict[str, float | int] | None = Field(
        None, description="Optional whitelisted training overrides: n_estimators, max_depth, learning_rate, seed, target_recall")


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to_version: str | None = None
    reason: str = Field("manual rollback", max_length=300)
