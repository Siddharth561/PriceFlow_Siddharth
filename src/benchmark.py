
import json
from pathlib import Path
import time

import numpy as np
import requests

URL = "http://localhost:8000/price"
PAYLOAD = {
    "lat": 18.52,
    "lng": 73.85,
    "time": "2026-10-10T18:30:00",
    "weather": "rain",
    "event": False,
}


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    latencies: list[float] = []
    with requests.Session() as session:
        for _ in range(20):
            response = session.post(URL, json=PAYLOAD, timeout=10)
            response.raise_for_status()
        for _ in range(1_000):
            started = time.perf_counter()
            response = session.post(URL, json=PAYLOAD, timeout=10)
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1_000)
    values = np.asarray(latencies)
    result = {
        "requests": len(values),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "p95_under_50ms": bool(np.percentile(values, 95) < 50),
    }
    print(
        "Latency ms: "
        f"mean={result['mean_ms']:.2f}, p50={result['p50_ms']:.2f}, "
        f"p95={result['p95_ms']:.2f}, p99={result['p99_ms']:.2f}, "
        f"max={result['max_ms']:.2f}; p95 < 50 ms: {result['p95_under_50ms']}"
    )
    with open("results/latency.json", "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2)


if __name__ == "__main__":
    main()
