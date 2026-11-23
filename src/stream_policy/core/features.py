from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


FEATURES = ("engagement", "financial", "activity", "authenticity")
POLICY_WEIGHTS = {
    "game": np.array([0.60, 0.19, 0.15, 0.06], dtype=float),
    "shop": np.array([0.24, 0.55, 0.15, 0.06], dtype=float),
    "shopping": np.array([0.24, 0.55, 0.15, 0.06], dtype=float),
    "other": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
    "mixed": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
}


def weights_for_context(context: str, retained: Sequence[str] | None = None) -> np.ndarray:
    base = POLICY_WEIGHTS.get(str(context), POLICY_WEIGHTS["other"])
    if retained is None:
        return base.copy()
    retained = tuple(retained)
    if not retained:
        raise ValueError("retained features must not be empty")
    indices = [FEATURES.index(name) for name in retained]
    selected = base[indices].astype(float, copy=True)
    total = float(selected.sum())
    return selected / total if total > 0.0 else np.full(len(indices), 1.0 / len(indices))


def validate_feature_vectors(values: np.ndarray, rows: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("feature vectors must have shape (n, 4)")
    if rows is not None and result.shape[0] != rows:
        raise ValueError("feature row count mismatch")
    if not np.isfinite(result).all() or (result < 0.0).any() or (result > 1.0).any():
        raise ValueError("feature vectors must be finite and lie in [0, 1]")
    return result


def validate_policy_weights(values: np.ndarray, width: int = 4) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size != width:
        raise ValueError("policy weights have an invalid shape")
    if not np.isfinite(result).all() or (result < 0.0).any() or not np.isclose(result.sum(), 1.0):
        raise ValueError("policy weights must be finite, nonnegative, and sum to one")
    return result


def normalized(values: np.ndarray, limits: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower = float(limits["min"])
    upper = float(limits["max"])
    median = limits["missing_median"]
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("invalid normalization limits")
    if median is not None:
        median = float(median)
        if not np.isfinite(median):
            raise ValueError("invalid normalization median")
        values = np.where(np.isfinite(values), values, median)
    if not np.isfinite(values).all():
        raise ValueError("normalization input contains non-finite values")
    if upper == lower:
        return np.zeros(values.shape, dtype=float)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def transform(raw: dict[str, np.ndarray], fit: dict) -> np.ndarray:
    required = ("watch_ms", "comment", "like", "gift_amount", "click", "age_log", "device_price_score")
    missing = set(required).difference(raw)
    if missing:
        raise KeyError(sorted(missing))
    arrays = {name: np.asarray(raw[name], dtype=float) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("raw feature lengths must match")
    gift = arrays["gift_amount"]
    if np.isinf(gift).any():
        raise ValueError("raw gift amounts must not contain infinity")
    arrays["gift_amount"] = np.where(np.isnan(gift), 0.0, gift)
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("raw features must be finite")
    if any((arrays[name] < 0.0).any() for name in ("watch_ms", "comment", "like", "gift_amount", "click")):
        raise ValueError("raw interaction features must be nonnegative")
    bounds = fit["columns"]
    watch = normalized(np.log1p(arrays["watch_ms"]), bounds["watch_log"])
    comment = normalized(arrays["comment"], bounds["comment"])
    like = normalized(arrays["like"], bounds["like"])
    financial = normalized(arrays["gift_amount"], bounds["gift_amount"])
    activity = np.minimum(arrays["click"] + arrays["like"] + arrays["comment"], float(fit["activity_p99"]))
    activity = normalized(activity, bounds["activity_capped"])
    age = normalized(arrays["age_log"], bounds["age_log"])
    device = normalized(arrays["device_price_score"], bounds["device_price_score"])
    result = np.stack([0.6 * watch + 0.2 * comment + 0.2 * like, financial, activity, 0.5 * age + 0.5 * device], axis=-1)
    return validate_feature_vectors(result)


def covariance(vectors: np.ndarray, ddof: int = 1) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim < 2 or vectors.shape[-2] <= ddof or vectors.shape[-1] == 0:
        raise ValueError("at least ddof plus one vectors are required")
    if not np.isfinite(vectors).all():
        raise ValueError("covariance input must be finite")
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    return centered.T @ centered / (vectors.shape[0] - ddof)


def rho(vectors: np.ndarray, reference: np.ndarray, ddof: int = 1, scale: float = 1.5) -> float:
    vectors = np.asarray(vectors, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("vectors must have shape (n, d)")
    if reference.shape != (vectors.shape[1], vectors.shape[1]) or not np.isfinite(reference).all():
        raise ValueError("reference covariance shape or values are invalid")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    distance = float(np.linalg.norm(covariance(vectors, ddof) - reference, ord="fro"))
    return float(np.clip(-np.expm1(-scale * distance), 0.0, 1.0))


def contribution(data: pd.DataFrame, retained: Sequence[str] = FEATURES) -> np.ndarray:
    retained = tuple(retained)
    missing = set(retained).difference(data.columns)
    if missing or "context" not in data:
        raise KeyError(sorted(missing | ({"context"} if "context" not in data else set())))
    values = data[list(retained)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("contribution features must be finite and lie in [0, 1]")
    result = np.zeros(len(data), dtype=float)
    contexts = data["context"].astype(str)
    for context in contexts.unique():
        mask = contexts.eq(context).to_numpy()
        result[mask] = values[mask] @ weights_for_context(context, retained)
    return result


def contiguous_history(data: pd.DataFrame) -> np.ndarray:
    required = {"streamer_id", "user_id", "cycle_id"}
    if not required.issubset(data.columns):
        raise KeyError(sorted(required.difference(data.columns)))
    if data.duplicated(["streamer_id", "user_id", "cycle_id"]).any():
        raise ValueError("duplicate canonical keys")
    ordered = data.assign(_row_position=np.arange(len(data))).sort_values(
        ["streamer_id", "user_id", "cycle_id"], kind="stable"
    )
    group_ids = ordered.groupby(["streamer_id", "user_id"], sort=False, observed=True).ngroup().to_numpy()
    cycles = pd.to_numeric(ordered["cycle_id"], errors="raise").to_numpy(dtype=float)
    same_contiguous = np.zeros(len(ordered), dtype=bool)
    if len(ordered) > 1:
        same_contiguous[1:] = (group_ids[1:] == group_ids[:-1]) & (cycles[1:] == cycles[:-1] + 1.0)
    breaks = np.cumsum(~same_contiguous)
    ordered_history = pd.Series(same_contiguous.astype(np.int8)).groupby(breaks, sort=False).cumsum().to_numpy(dtype=np.int8)
    result = np.empty(len(data), dtype=np.int8)
    result[ordered["_row_position"].to_numpy(dtype=int)] = np.minimum(ordered_history, 5).astype(np.int8)
    return result


def fit_covariance_reference(data: pd.DataFrame) -> dict[str, np.ndarray | int]:
    required = {"split", *FEATURES}
    if not required.issubset(data.columns):
        raise KeyError(sorted(required.difference(data.columns)))
    fit = data["split"].eq("fit").to_numpy() & data[list(FEATURES)].notna().all(axis=1).to_numpy()
    matrix = data.loc[fit, list(FEATURES)].to_numpy(dtype=float)
    if len(matrix) < 2:
        raise ValueError("at least two fit rows are required")
    validate_feature_vectors(matrix)
    return {"4d": covariance(matrix), "3d_no_auth": covariance(matrix[:, :3]), "fit_rows": int(len(matrix))}


def reliability_from_history(
    data: pd.DataFrame,
    reference: np.ndarray,
    columns: Sequence[str] = FEATURES,
    history: np.ndarray | None = None,
    scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    columns = tuple(columns)
    if not columns or set(columns).difference(FEATURES):
        raise ValueError("reliability columns must be declared features")
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (len(columns), len(columns)) or not np.isfinite(reference).all():
        raise ValueError("reliability reference shape or values are invalid")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    required = {"streamer_id", "user_id", "cycle_id", *columns}
    if not required.issubset(data.columns):
        raise KeyError(sorted(required.difference(data.columns)))
    if data.duplicated(["streamer_id", "user_id", "cycle_id"]).any():
        raise ValueError("duplicate canonical keys")
    order = data.assign(_row_position=np.arange(len(data))).sort_values(
        ["streamer_id", "user_id", "cycle_id"], kind="stable"
    )
    ordered = order.reset_index(drop=True)
    if history is None:
        ordered_history = contiguous_history(ordered)
    else:
        supplied = np.asarray(history)
        if supplied.shape != (len(data),):
            raise ValueError("history shape mismatch")
        ordered_history = supplied[order["_row_position"].to_numpy(dtype=int)]
    groups = ordered.groupby(["streamer_id", "user_id"], sort=False, observed=True)
    cycles = pd.to_numeric(ordered["cycle_id"], errors="raise").to_numpy(dtype=float)
    candidates = np.asarray(ordered_history) >= 5
    lag_cycle = np.column_stack([groups["cycle_id"].shift(lag).to_numpy(dtype=float) for lag in range(1, 6)])
    candidates &= np.isfinite(lag_cycle).all(axis=1)
    candidates &= np.all(cycles[:, None] - lag_cycle == np.arange(1, 6, dtype=float), axis=1)
    lag_values = []
    for column in columns:
        values = np.column_stack([groups[column].shift(lag).to_numpy(dtype=float) for lag in range(1, 6)])
        lag_values.append(values)
        candidates &= np.isfinite(values).all(axis=1)
    ordered_scores = np.full(len(ordered), np.nan, dtype=float)
    if candidates.any():
        tensor = np.stack([values[candidates] for values in lag_values], axis=2)
        centered = tensor - tensor.mean(axis=1, keepdims=True)
        covariance_values = np.einsum("nki,nkj->nij", centered, centered) / 4.0
        distance = np.linalg.norm(covariance_values - reference, axis=(1, 2))
        ordered_scores[candidates] = np.clip(-np.expm1(-scale * distance), 0.0, 1.0)
    scores = np.empty(len(data), dtype=float)
    positions = order["_row_position"].to_numpy(dtype=int)
    scores[positions] = ordered_scores
    scored = np.zeros(len(data), dtype=bool)
    scored[positions] = candidates
    return scores, scored
