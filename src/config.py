
import numpy as np

SURGE_LEVELS = np.array(
    [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    dtype=np.float32,
)
LAT_MIN = 18.43
LAT_MAX = 18.63
LNG_MIN = 73.75
LNG_MAX = 73.95
GRID = 5
BASE_PRICE = 150.0
PEAK_HOURS = [8, 9, 10, 17, 18, 19, 20]
WEATHER = {"clear": 0, "cloudy": 1, "rain": 2}
MAX_DEMAND = 250.0
MAX_SUPPLY = 60.0
