from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from config import LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, MAX_BASE_PRICE, MAX_DEMAND, MAX_SUPPLY


class PriceRequest(BaseModel):
    """Request body for a pricing quote based on a market snapshot."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=LAT_MIN, le=LAT_MAX, allow_inf_nan=False)
    lng: float = Field(..., ge=LNG_MIN, le=LNG_MAX, allow_inf_nan=False)
    hour: int = Field(..., strict=True, ge=0, le=23)
    day_of_week: int = Field(..., strict=True, ge=0, le=6)
    weather: Literal["clear", "cloudy", "rain"]
    demand: float = Field(..., ge=15, le=MAX_DEMAND, allow_inf_nan=False)
    supply: float = Field(..., ge=6, le=MAX_SUPPLY, allow_inf_nan=False)
    base_price: float = Field(..., ge=60, le=MAX_BASE_PRICE, allow_inf_nan=False)


class PriceResponse(BaseModel):

    surge_multiplier: float
    base_price: float
    final_price: float
    estimated_acceptance_probability: float
    model_identifier: str
    metadata: dict[str, Any]
    inference_latency_ms: float
    database_latency_ms: float | None = None
    logged_to_mongodb: bool = False
