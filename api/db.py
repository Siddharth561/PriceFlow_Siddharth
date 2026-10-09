
import os
from typing import Any

from pymongo import MongoClient

_client: MongoClient | None = None
_collection: Any = None


def connect() -> None:
    global _client, _collection
    try:
        _client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=500)
        _client.admin.command("ping")
        _collection = _client["priceflow"]["pricing_logs"]
    except Exception:
        _collection = None


def log_decision(doc: dict[str, Any]) -> None:
    try:
        if _collection is not None:
            _collection.insert_one(doc)
    except Exception:
        pass
