"""
NexGuard — Inference Benchmarking Engine
========================================
Measures real-world inference latency and frame processing throughput:
- Warm-up iterations to stabilize PyTorch / OpenCV runtime
- Percentile latency statistics: P50, P90, P95, P99, Mean, Min, Max
- Frames-Per-Second (FPS) measurement under sustained execution
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO

@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    resolution: int
    num_iterations: int
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float

class InferenceBenchmark:
    """
    Benchmarks real-time inference latency and FPS.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)

    def run(
        self,
        imgsz: int = 224,
        warmup: int = 10,
        iterations: int = 50,
    ) -> BenchmarkResult:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            self.model(dummy, verbose=False, device=self.device)

        # Timed iterations
        latencies: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.model(dummy, verbose=False, device=self.device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return BenchmarkResult(
            model_name=self.model_path,
            device=self.device,
            resolution=imgsz,
            num_iterations=iterations,
            mean_latency_ms=round(mean_lat, 2),
            p50_latency_ms=round(p50, 2),
            p90_latency_ms=round(p90, 2),
            p95_latency_ms=round(p95, 2),
            p99_latency_ms=round(p99, 2),
            fps=round(fps, 1),
        )
