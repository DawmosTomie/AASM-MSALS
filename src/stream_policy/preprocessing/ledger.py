from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CANONICAL_KEYS = ("streamer_id", "user_id", "cycle_id")
EVENT_COLUMNS = ("streamer_id", "user_id", "timestamp", "click", "comment", "like", "gift")


def device_price_mapping(values: Iterable[object], fit_median: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(float(fit_median)):
        raise ValueError("fit_median must be finite")
    mapped = []
    unknown = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = np.nan
        if not np.isfinite(number) or number <= 0:
            mapped.append(float(fit_median))
            unknown.append(True)
        elif number < 1000:
            mapped.append(500.5)
            unknown.append(False)
        elif number < 2000:
            mapped.append(1500.0)
            unknown.append(False)
        elif number < 4000:
            mapped.append(3000.0)
            unknown.append(False)
        elif number < 5000:
            mapped.append(4500.0)
            unknown.append(False)
        else:
            mapped.append(5000.0)
            unknown.append(False)
    return np.asarray(mapped, dtype=float), np.asarray(unknown, dtype=bool)


def fit_minmax(values: Iterable[float], fit_mask: Iterable[bool]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    mask = np.asarray(list(fit_mask), dtype=bool)
    if data.shape != mask.shape:
        raise ValueError("fit_mask length mismatch")
    finite = data[mask & np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("no finite fit values")
    return {"min": float(finite.min()), "max": float(finite.max()), "missing_median": float(np.median(finite))}


def fit_p99(values: Iterable[float], fit_mask: Iterable[bool]) -> float:
    data = np.asarray(list(values), dtype=float)
    mask = np.asarray(list(fit_mask), dtype=bool)
    if data.shape != mask.shape:
        raise ValueError("fit_mask length mismatch")
    finite = data[mask & np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("no finite fit values")
    return float(np.quantile(finite, 0.99))


def assign_locked_split(cycle_id: Iterable[int]) -> np.ndarray:
    cycles = np.asarray(list(cycle_id), dtype=int)
    if np.any(cycles < 1) or np.any(cycles > 21):
        raise ValueError("cycle_id must be in 1..21")
    return np.select([cycles <= 12, cycles <= 16], ["fit", "validation"], default="test")


def _required(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(missing)


def validate_canonical_rows(frame: pd.DataFrame) -> None:
    _required(frame, CANONICAL_KEYS)
    if frame[list(CANONICAL_KEYS)].isna().any().any():
        raise ValueError("canonical keys must be non-null")
    if frame.duplicated(list(CANONICAL_KEYS)).any():
        raise ValueError("duplicate canonical keys")
    cycles = pd.to_numeric(frame["cycle_id"], errors="raise")
    if not np.equal(cycles, cycles.astype(int)).all() or (cycles < 1).any() or (cycles > 21).any():
        raise ValueError("cycle_id must be an integer in 1..21")


def _local_timestamp(values: pd.Series, timezone: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("unusable timestamps")
    return parsed.dt.tz_convert(timezone)


def canonicalize_events(
    events: pd.DataFrame,
    start: str | pd.Timestamp,
    timezone: str = "Asia/Shanghai",
    impossible_mask: pd.Series | None = None,
) -> pd.DataFrame:
    _required(events, EVENT_COLUMNS)
    data = events.copy()
    data["timestamp"] = _local_timestamp(data["timestamp"], timezone)
    start_local = pd.Timestamp(start)
    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(timezone)
    else:
        start_local = start_local.tz_convert(timezone)
    end_local = start_local + pd.Timedelta(days=21)
    interaction = data[["click", "comment", "like", "gift"]].apply(pd.to_numeric, errors="coerce")
    valid = data["streamer_id"].notna() & data["user_id"].notna()
    valid &= data["timestamp"].ge(start_local) & data["timestamp"].lt(end_local)
    valid &= data["user_id"].ne(data["streamer_id"])
    valid &= interaction.notna().all(axis=1) & interaction.gt(0).any(axis=1)
    if impossible_mask is not None:
        if len(impossible_mask) != len(data):
            raise ValueError("impossible_mask length mismatch")
        valid &= ~np.asarray(impossible_mask, dtype=bool)
    data = data.loc[valid].copy()
    data[["click", "comment", "like", "gift"]] = interaction.loc[valid]
    delta = data["timestamp"] - start_local
    data["cycle_id"] = (delta.dt.total_seconds() // (24 * 3600)).astype(int) + 1
    data["split"] = assign_locked_split(data["cycle_id"])
    aggregate = {column: "sum" for column in ("click", "comment", "like", "gift")}
    if "watch_ms" in data:
        aggregate["watch_ms"] = "sum"
    if "gift_amount" in data:
        aggregate["gift_amount"] = "sum"
    if "context" in data:
        aggregate["context"] = lambda values: sorted(values.astype(str))[len(values) // 2]
    result = data.groupby(list(CANONICAL_KEYS), sort=True, observed=True).agg(aggregate).reset_index()
    result["split"] = assign_locked_split(result["cycle_id"])
    validate_canonical_rows(result)
    return result
