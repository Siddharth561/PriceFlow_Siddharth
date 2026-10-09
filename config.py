from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

WEATHER_INDEX = {"clear": 0, "cloudy": 1, "rain": 2}
WEATHER_LABELS = ["clear", "cloudy", "rain"]
SURGE_LEVELS = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], dtype=np.float32)
LAT_MIN = 18.43
LAT_MAX = 18.63
LNG_MIN = 73.75
LNG_MAX = 73.95
GRID_SIZE = 5
BASE_PRICE = 150.0
EPISODE_HORIZON = 24
TRAINING_EPISODES_TARGET = 10_000
MAX_DEMAND = 220.0
MAX_SUPPLY = 110.0
MAX_BASE_PRICE = 500.0
DEFAULT_MODEL_PATH = MODELS_DIR / "dqn_pricing.zip"
DEFAULT_DATA_PATH = DATA_DIR / "rides_trips.csv"

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "priceflow")
