from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

logger = logging.getLogger(__name__)
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "priceflow")

_client: MongoClient[Any] | None = None
_decision_store: MongoDecisionStore | None = None


def get_database() -> Any:

    global _client

    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is missing from the local environment.")
    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _client[MONGODB_DATABASE]


def close_connection() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


class MongoDecisionStore:

    def __init__(self) -> None:
        self.collection: Collection[dict[str, Any]] | None = None
        self.connected = False

    def connect(self) -> bool:
        """Ping MongoDB and select ``pricing_logs``; disable logging on failure."""
        try:
            database = get_database()
            database.client.admin.command("ping")
            self.collection = database["pricing_logs"]
            self.connected = True
        except (PyMongoError, RuntimeError) as exc:
            self.collection = None
            self.connected = False
            logger.warning(
                "MongoDB pricing logging is unavailable (%s).",
                type(exc).__name__,
            )
        return self.connected

    def check_connection(self) -> bool:
        return self.connect()

    def persist(self, document: dict[str, Any]) -> tuple[bool, float | None]:
        """Insert one pricing record and return success plus database latency."""
        if not self.connected or self.collection is None:
            if not self.connect():
                return False, None

        started_at = time.perf_counter()
        try:
            if self.collection is None:
                self.connected = False
                return False, None
            self.collection.insert_one(document)
        except PyMongoError as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.connected = False
            self.collection = None
            logger.warning(
                "MongoDB pricing log insert failed (%s); returning the pricing decision without persistence.",
                type(exc).__name__,
            )
            return False, elapsed_ms
        return True, (time.perf_counter() - started_at) * 1000.0

    def close(self) -> None:
        """Release MongoDB resources on application shutdown."""
        self.collection = None
        self.connected = False
        close_connection()


def get_decision_store() -> MongoDecisionStore:
    global _decision_store
    if _decision_store is None:
        _decision_store = MongoDecisionStore()
    return _decision_store
