from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.features import covariance, rho
from ..attacks.windows import traffic_score
from .metrics import confusion, select_validation_threshold


@dataclass
class WindowBatch:
    metadata: pd.DataFrame
    raw: dict[str, np.ndarray]
    features: np.ndarray

    def take(self, selector: object) -> "WindowBatch":
        return WindowBatch(
            self.metadata.iloc[selector].reset_index(drop=True),
            {name: np.asarray(values)[selector].copy() for name, values in self.raw.items()},
            np.asarray(self.features)[selector].copy(),
        )


def window_summary(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("window vectors must be a finite (n, history, d) array")
    return np.concatenate([values.mean(axis=1), np.ptp(values, axis=1)], axis=1)


def gini_reduction(values: Iterable[float], groups: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(list(values), dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError("gini values must be finite, nonnegative and nonempty")
    if not isinstance(groups, pd.DataFrame) or len(groups) != len(scores) or groups.shape[1] == 0:
        raise ValueError("groups must be a nonempty DataFrame matching values")
    table = groups.reset_index(drop=True).copy()
    keys = list(table.columns)
    table["position"] = np.arange(len(scores))
    table["value"] = scores
    table.sort_values([*keys, "value", "position"], inplace=True, kind="stable")
    peers = table.groupby(keys, sort=False, observed=True, dropna=False)
    table["rank"] = peers.cumcount() + 1
    table["weighted"] = table["rank"] * table["value"]
    peers = table.groupby(keys, sort=False, observed=True, dropna=False)
    n = peers["value"].transform("size").to_numpy(dtype=float)
    total = peers["value"].transform("sum").to_numpy(dtype=float)
    running = peers["value"].cumsum().to_numpy(dtype=float)
    weighted = peers["weighted"].transform("sum").to_numpy(dtype=float)
    remaining = total - table["value"].to_numpy(dtype=float)
    original = np.divide(2.0 * weighted, n * total, out=np.zeros(len(table)), where=total > 0.0)
    original = np.where(total > 0.0, original - (n + 1.0) / n, 0.0)
    deletion = weighted - table["weighted"].to_numpy(dtype=float) - (total - running)
    after = np.divide(2.0 * deletion, (n - 1.0) * remaining, out=np.zeros(len(table)), where=(n > 1.0) & (remaining > 0.0))
    after = np.where((n > 1.0) & (remaining > 0.0), after - n / np.maximum(n - 1.0, 1.0), 0.0)
    result = np.empty(len(table), dtype=float)
    result[table["position"].to_numpy(dtype=int)] = original - after
    return result


def traffic_evidence(raw: dict[str, np.ndarray]) -> np.ndarray:
    required = {"click", "watch_ms"}
    if not required.issubset(raw):
        raise KeyError(sorted(required.difference(raw)))
    values = {key: np.asarray(raw[key], dtype=float) for key in required}
    if values["click"].shape != values["watch_ms"].shape or not np.isfinite(values["click"]).all() or not np.isfinite(values["watch_ms"]).all() or (values["click"] < 0.0).any() or (values["watch_ms"] < 0.0).any():
        raise ValueError("traffic evidence must be finite, nonnegative and shape-aligned")
    return traffic_score(values)


def validation_threshold(scores: Iterable[float], labels: Iterable[bool], grid: Iterable[float]) -> float:
    return select_validation_threshold(labels, scores, grid)


def detector_window_scores(vectors: np.ndarray, reference: np.ndarray, ddof: int = 1, scale: float = 1.5) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 3 or values.shape[1] <= ddof or not np.isfinite(values).all():
        raise ValueError("detector windows must have shape (n, history, d) with finite values")
    return np.asarray([rho(window, reference, ddof, scale) for window in values], dtype=float)


__all__ = [
    "WindowBatch",
    "window_summary",
    "traffic_evidence",
    "detector_window_scores",
    "gini_reduction",
    "confusion",
    "validation_threshold",
    "covariance",
]
