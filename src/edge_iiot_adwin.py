from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable

import numpy as np


@dataclass
class ADWINState:
    delta: float = 0.002
    min_window: int = 20
    max_window: int = 4096
    window: Deque[float] = field(default_factory=deque)
    drift_count: int = 0
    last_drift_index: int | None = None
    total_seen: int = 0


class ADWINMonitor:
    def __init__(self, delta: float = 0.002, min_window: int = 20, max_window: int = 4096) -> None:
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1).")
        self.state = ADWINState(delta=delta, min_window=min_window, max_window=max_window)

    @property
    def delta(self) -> float:
        return self.state.delta

    def reset(self) -> None:
        self.state.window.clear()
        self.state.drift_count = 0
        self.state.last_drift_index = None
        self.state.total_seen = 0

    def _epsilon(self, n0: int, n1: int, variance: float) -> float:
        if n0 <= 0 or n1 <= 0:
            return math.inf
        n = n0 + n1
        m = (1.0 / n0) + (1.0 / n1)
        variance = max(variance, 1e-12)
        first = math.sqrt(2.0 * variance * math.log(2.0 / self.state.delta) * m)
        second = (2.0 / 3.0) * math.log(2.0 / self.state.delta) * m
        return first + second

    def update(self, value: float) -> dict[str, object]:
        val = float(value)
        self.state.window.append(val)
        self.state.total_seen += 1
        if len(self.state.window) > self.state.max_window:
            self.state.window.popleft()

        drift_detected = False
        cut_index = None
        mean_before = None
        mean_after = None

        values = np.asarray(self.state.window, dtype=float)
        if len(values) >= self.state.min_window * 2:
            variance = float(np.var(values))
            for cut in range(self.state.min_window, len(values) - self.state.min_window + 1):
                old = values[:cut]
                new = values[cut:]
                diff = abs(float(old.mean()) - float(new.mean()))
                eps = self._epsilon(len(old), len(new), variance)
                if diff > eps:
                    drift_detected = True
                    cut_index = cut
                    mean_before = float(old.mean())
                    mean_after = float(new.mean())
                    self.state.window = deque(new.tolist())
                    self.state.drift_count += 1
                    self.state.last_drift_index = self.state.total_seen
                    break

        return {
            "drift_detected": drift_detected,
            "cut_index": cut_index,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "window_width": int(len(self.state.window)),
            "running_mean": float(np.mean(self.state.window)) if self.state.window else 0.0,
            "running_std": float(np.std(self.state.window)) if self.state.window else 0.0,
            "drift_count": int(self.state.drift_count),
            "last_drift_index": self.state.last_drift_index,
            "delta": float(self.state.delta),
        }

    def update_many(self, values: Iterable[float]) -> list[dict[str, object]]:
        return [self.update(value) for value in values]

