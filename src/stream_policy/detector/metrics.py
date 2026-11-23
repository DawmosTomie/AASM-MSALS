from __future__ import annotations

from typing import Iterable

import numpy as np

from ..core.features import rho
from ..attacks.windows import traffic_score


def gini_reduction(values: Iterable[float], groups: object) -> np.ndarray:
    from .evidence import gini_reduction as _gini_reduction

    return _gini_reduction(values, groups)


def confusion(labels: Iterable[bool], predictions: Iterable[bool]) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=bool)
    p = np.asarray(list(predictions), dtype=bool)
    if y.shape != p.shape:
        raise ValueError("labels and predictions must have equal length")
    tp = int(np.count_nonzero(y & p))
    fp = int(np.count_nonzero(~y & p))
    tn = int(np.count_nonzero(~y & ~p))
    fn = int(np.count_nonzero(y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if tn else 0.0
    macro_f1 = 0.5 * (f1 + negative_f1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "n": int(y.size), "prevalence": float(y.mean()) if y.size else 0.0, "prediction_rate": float(p.mean()) if y.size else 0.0, "precision": precision, "recall": recall, "f1": f1, "f1_pos": f1, "f1_neg": negative_f1, "macro_f1": macro_f1, "fpr": fpr}


def select_validation_threshold(labels: Iterable[bool], scores: Iterable[float], grid: Iterable[float] | None = None) -> float:
    y = np.asarray(list(labels), dtype=bool)
    s = np.asarray(list(scores), dtype=float)
    if y.shape != s.shape or y.size == 0 or not np.isfinite(s).all():
        raise ValueError("invalid validation vectors")
    candidates = np.linspace(0.0, 1.0, 101) if grid is None else np.asarray(list(grid), dtype=float)
    if candidates.size == 0 or not np.isfinite(candidates).all():
        raise ValueError("invalid threshold grid")
    best = None
    for threshold in candidates:
        stats = confusion(y, s > threshold)
        key = (float(stats["macro_f1"]), -float(stats["fp"]), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def evaluate(labels: Iterable[bool], scores: Iterable[float], threshold: float) -> dict[str, float | int]:
    if type(threshold) not in (int, float) or not np.isfinite(threshold):
        raise TypeError("threshold must be finite")
    return confusion(labels, np.asarray(list(scores), dtype=float) > float(threshold))


def detector_scores(data: object, reference4: np.ndarray, reference3: np.ndarray, cohort: str, ddof: int = 1, scale: float = 1.5) -> dict[str, np.ndarray]:
    from .evidence import detector_window_scores

    features = np.asarray(getattr(data, "features"), dtype=float)
    if features.ndim != 3 or features.shape[-1] != 4 or features.shape[-2] < ddof + 1:
        raise ValueError("detector windows must have shape (n, history, 4)")
    metadata = getattr(data, "metadata")
    raw = getattr(data, "raw")
    grouping = ["streamer_id", "cycle_id"] if cohort == "actor" else ["context", "cycle_id"]
    if not set(grouping).issubset(metadata.columns):
        raise KeyError(grouping)
    return {"Proposed": detector_window_scores(features, reference4, ddof, scale), "No-Authenticity": detector_window_scores(features[..., :3], reference3, ddof, scale), "Univariate": features[:, -1, 2], "Static-Gini": gini_reduction(traffic_score(raw)[:, -1], metadata[grouping])}
