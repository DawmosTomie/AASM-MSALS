from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.features import transform


RAW = ("click", "watch_ms", "comment", "like", "gift_amount", "age_log", "device_price_score")


@dataclass
class Windows:
    metadata: pd.DataFrame
    raw: dict[str, np.ndarray]
    features: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, pd.DataFrame):
            raise TypeError("metadata must be a pandas DataFrame")
        values = np.asarray(self.features)
        if values.ndim != 3 or values.shape[0] != len(self.metadata) or values.shape[2] != 4:
            raise ValueError("features must have shape (rows, history, 4)")
        if not self.raw or set(self.raw) != set(RAW):
            raise ValueError("raw must contain the locked raw feature names")
        for name in RAW:
            array = np.asarray(self.raw[name])
            if array.ndim != 2 or array.shape[:2] != values.shape[:2]:
                raise ValueError("raw features must have shape (rows, history)")
            if not np.isfinite(array).all():
                raise ValueError("raw features must be finite")

    @property
    def history(self) -> int:
        return int(self.features.shape[1])

    def take(self, selector: object) -> "Windows":
        selected = self.metadata.iloc[selector].reset_index(drop=True)
        return Windows(selected, {name: values[selector].copy() for name, values in self.raw.items()}, self.features[selector].copy())


def traffic_score(raw: dict[str, np.ndarray]) -> np.ndarray:
    values = {name: np.asarray(raw[name], dtype=float) for name in ("click", "watch_ms")}
    if values["click"].shape != values["watch_ms"].shape:
        raise ValueError("click and watch_ms shapes must match")
    return 0.5 * values["click"] + 0.5 * values["watch_ms"] / 1000.0
