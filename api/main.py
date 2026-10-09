from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status

from api.database import MongoDecisionStore, get_decision_store
from api.pricing_service import PricingService, get_pricing_service
from api.schemas import PriceRequest, PriceResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the DQN and connect to MongoDB once for the process lifetime."""
    service = get_pricing_service()
    store = get_decision_store()
    app.state.model_loaded = False
    app.state.model_error = None

    try:
        service.load_model()
        app.state.model_loaded = True
    except Exception as exc:
        app.state.model_error = type(exc).__name__
        logger.exception("Pricing model failed to load; /price will return 503.")

    app.state.database_connected = store.connect()
    try:
        yield
    finally:
        store.close()
        service.close()
        app.state.model_loaded = False


app = FastAPI(title="PriceFlow Pricing API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    """Return a lightweight API descriptor."""
    return {"service": "PriceFlow", "status": "ready"}


@app.get("/health")
def health() -> dict[str, bool | str]:
    model_loaded = bool(getattr(app.state, "model_loaded", False))
    store = get_decision_store()
    database_connected = store.check_connection()
    app.state.database_connected = database_connected
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "database_connected": database_connected,
    }


@app.post("/price", response_model=PriceResponse)
def price(request: PriceRequest) -> PriceResponse:
    """Return a model quote and persist it to MongoDB when available."""
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Pricing model is unavailable. Confirm that models/dqn_pricing.zip "
                "exists and was trained for the current environment."
            ),
        )

    service: PricingService = get_pricing_service()
    store: MongoDecisionStore = get_decision_store()
    try:
        decision = service.predict(request)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Pricing inference is unavailable (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing model is unavailable; verify the trained model artifact.",
        ) from exc

    log_document = {
        "request": request.model_dump(),
        "decision": {
            "surge_multiplier": decision.surge_multiplier,
            "base_price": decision.base_price,
            "final_price": decision.final_price,
            "estimated_acceptance_probability": decision.estimated_acceptance_probability,
            "model_identifier": decision.model_identifier,
        },
        "inference_latency_ms": decision.inference_latency_ms,
        "created_at": datetime.now(timezone.utc),
    }
    logged, database_latency_ms = store.persist(log_document)
    app.state.database_connected = store.connected
    decision.database_latency_ms = database_latency_ms
    decision.logged_to_mongodb = logged
    return decision
