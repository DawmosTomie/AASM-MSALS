from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..core.features import (
    FEATURES,
    contribution,
    reliability_from_history,
    validate_policy_weights,
    weights_for_context,
)


CANONICAL_KEYS = ("streamer_id", "user_id", "cycle_id")
TENANT_KEYS = ("streamer_id", "cycle_id")


def validate_canonical_keys(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("canonical rows must be a pandas DataFrame")
    missing = set(CANONICAL_KEYS).difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    if frame[list(CANONICAL_KEYS)].isna().any().any():
        raise ValueError("canonical keys must be non-null")
    if frame.duplicated(list(CANONICAL_KEYS)).any():
        raise ValueError("duplicate canonical keys")


def weighted_contribution(frame: pd.DataFrame, retained: Iterable[str] = FEATURES) -> np.ndarray:
    return contribution(frame, tuple(retained))


def gini(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0.0).any():
        raise ValueError("allocation values must be finite and nonnegative")
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return np.nan
    x = np.sort(x)
    n = x.size
    return float(2.0 * np.dot(np.arange(1, n + 1), x) / (n * x.sum()) - (n + 1) / n)


def bottom_half_share(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0.0).any():
        raise ValueError("allocation values must be finite and nonnegative")
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return np.nan
    x = np.sort(x)
    cutoff = int(np.floor(x.size * 0.5))
    return float(x[:cutoff].sum() / x.sum()) if cutoff else 0.0


def per_cycle_metrics(frame: pd.DataFrame, amount_col: str = "allocation") -> pd.DataFrame:
    required = set(TENANT_KEYS) | {amount_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    rows: list[dict[str, Any]] = []
    for (streamer_id, cycle_id), group in frame.groupby(list(TENANT_KEYS), sort=False, observed=True):
        amounts = group[amount_col].to_numpy(dtype=float)
        total = float(amounts.sum())
        rows.append({
            "streamer_id": streamer_id,
            "cycle_id": cycle_id,
            "n_recipients": int(len(group)),
            "total_paid": total,
            "gini": gini(amounts),
            "bottom_half_share": bottom_half_share(amounts),
            "zero_payout": bool(np.isclose(total, 0.0)),
        })
    return pd.DataFrame(rows)


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _vector(values: Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{name} has an invalid shape")
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def allocate_proportional(adjusted_contribution: Iterable[float], budget_available: float) -> np.ndarray:
    values = np.asarray(list(adjusted_contribution), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("adjusted contributions must be finite and nonnegative")
    budget = _nonnegative_finite(budget_available, "budget_available")
    allocation = np.zeros(values.shape, dtype=float)
    total = float(values.sum())
    if values.size == 0 or np.isclose(total, 0.0):
        return allocation
    allocation = budget * values / total
    allocation[int(np.argmax(values))] += budget - float(allocation.sum())
    if np.any(allocation < -np.finfo(float).eps) or not np.isclose(allocation.sum(), budget, rtol=0.0, atol=1e-12):
        raise ArithmeticError("proportional allocation failed to conserve budget")
    return np.maximum(allocation, 0.0)


def allocate_scores(frame: pd.DataFrame, score_col: str, budget_col: str = "common_available_budget") -> pd.DataFrame:
    validate_canonical_keys(frame)
    if score_col not in frame or budget_col not in frame:
        raise KeyError(sorted({score_col, budget_col}.difference(frame.columns)))
    result = frame.copy()
    result[score_col] = pd.to_numeric(result[score_col], errors="coerce")
    result[budget_col] = pd.to_numeric(result[budget_col], errors="coerce")
    if not np.isfinite(result[score_col].to_numpy(dtype=float)).all() or (result[score_col] < 0.0).any():
        raise ValueError("scores must be finite and nonnegative")
    if not np.isfinite(result[budget_col].to_numpy(dtype=float)).all() or (result[budget_col] < 0.0).any():
        raise ValueError("budgets must be finite and nonnegative")
    result["allocation"] = 0.0
    result["deferred_budget_actual"] = 0.0
    for _, index in result.groupby(list(TENANT_KEYS), sort=False, observed=True).groups.items():
        positions = result.index.get_indexer(index)
        budgets = result.iloc[positions][budget_col].to_numpy(dtype=float)
        if not np.allclose(budgets, budgets[0], rtol=0.0, atol=0.0):
            raise ValueError("one common budget is required per tenant-cycle")
        values = result.iloc[positions][score_col].to_numpy(dtype=float)
        result.iloc[positions, result.columns.get_loc("allocation")] = allocate_proportional(values, float(budgets[0]))
        if np.isclose(values.sum(), 0.0):
            result.iloc[positions, result.columns.get_loc("deferred_budget_actual")] = float(budgets[0])
    return result


def attach_carry_forward(frame: pd.DataFrame, budget: float = 1000.0) -> pd.DataFrame:
    validate_canonical_keys(frame)
    if "raw_traffic" not in frame:
        raise KeyError("raw_traffic")
    budget = _nonnegative_finite(budget, "budget")
    traffic = pd.to_numeric(frame["raw_traffic"], errors="coerce")
    if not np.isfinite(traffic.to_numpy(dtype=float)).all() or (traffic < 0.0).any():
        raise ValueError("raw_traffic must be finite and nonnegative")
    cycle = frame[list(TENANT_KEYS)].copy()
    cycle["raw_total"] = traffic.to_numpy(dtype=float)
    cycle = cycle.groupby(list(TENANT_KEYS), observed=True, sort=True)["raw_total"].sum().reset_index()
    cycle = cycle.sort_values(list(TENANT_KEYS), kind="stable").reset_index(drop=True)
    cycle["new_budget"] = cycle.groupby("streamer_id", sort=False).cumcount().add(1).astype(float) * budget
    cycle["common_four_arm_paid"] = cycle["raw_total"].gt(0.0)
    settled = np.where(cycle["common_four_arm_paid"], cycle["new_budget"], 0.0)
    cycle["settled"] = pd.Series(settled).groupby(cycle["streamer_id"], sort=False).cummax().to_numpy()
    cycle["before"] = cycle.groupby("streamer_id", sort=False)["settled"].shift(1, fill_value=0.0)
    cycle["common_available_budget"] = cycle["new_budget"] - cycle["before"]
    cycle["deferred_budget"] = np.where(cycle["common_four_arm_paid"], 0.0, cycle["common_available_budget"])
    return frame.merge(cycle[list(TENANT_KEYS) + ["common_available_budget", "common_four_arm_paid", "deferred_budget"]], on=list(TENANT_KEYS), how="left", validate="many_to_one")


def _contributions(frame: pd.DataFrame, retained: tuple[str, ...], policy_weights: Any) -> np.ndarray:
    missing = set(retained).difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    if policy_weights is None:
        return weighted_contribution(frame, retained)
    values = frame[list(retained)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("contribution features must be finite and lie in [0, 1]")
    if isinstance(policy_weights, Mapping):
        if "context" not in frame:
            raise KeyError("context")
        result = np.zeros(len(frame), dtype=float)
        contexts = frame["context"].astype(str)
        for context in contexts.unique():
            if context not in policy_weights:
                raise KeyError(context)
            weights = validate_policy_weights(policy_weights[context], len(retained))
            mask = contexts.eq(context).to_numpy()
            result[mask] = values[mask] @ weights
        return result
    weights = validate_policy_weights(policy_weights, len(retained))
    return values @ weights


def _reliability(
    frame: pd.DataFrame,
    rho_values: Iterable[float] | None,
    covariance_reference: np.ndarray | None,
    history: np.ndarray | None,
    unscored_rho: float | None,
    retained: tuple[str, ...],
    reliability_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=bool)
    if rho_values is not None and covariance_reference is not None:
        raise ValueError("supply rho_values or covariance_reference, not both")
    if rho_values is not None:
        values = np.asarray(list(rho_values), dtype=float)
        if values.shape != (len(frame),):
            raise ValueError("rho_values shape mismatch")
        scored = np.isfinite(values)
        if not scored.all():
            if unscored_rho is None:
                raise ValueError("unscored reliability requires an explicit payout fallback")
            fallback = _nonnegative_finite(unscored_rho, "unscored_rho")
            if fallback > 1.0:
                raise ValueError("unscored_rho must lie in [0, 1]")
            values = np.where(scored, values, fallback)
    elif covariance_reference is not None:
        values, scored = reliability_from_history(
            frame,
            covariance_reference,
            columns=retained,
            history=history,
            scale=reliability_scale,
        )
        if not scored.all():
            if unscored_rho is None:
                raise ValueError("incomplete five-cycle history is unscored and needs an explicit payout fallback")
            fallback = _nonnegative_finite(unscored_rho, "unscored_rho")
            if fallback > 1.0:
                raise ValueError("unscored_rho must lie in [0, 1]")
            values = np.where(scored, values, fallback)
    else:
        raise ValueError("continuous reliability input is required")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("rho values must lie in [0, 1]")
    return values, scored


def execute_cau(
    frame: pd.DataFrame,
    new_budget: float,
    *,
    deferred_budget: float = 0.0,
    policy_weights: Any = None,
    rho_values: Iterable[float] | None = None,
    covariance_reference: np.ndarray | None = None,
    history: np.ndarray | None = None,
    unscored_rho: float | None = None,
    retained: Iterable[str] = FEATURES,
    reliability_scale: float = 1.5,
) -> dict[str, Any]:
    validate_canonical_keys(frame)
    if len(frame) and (frame["streamer_id"].nunique(dropna=False) != 1 or frame["cycle_id"].nunique(dropna=False) != 1):
        raise ValueError("execute_cau requires one tenant-cycle")
    retained = tuple(retained)
    if not retained or set(retained).difference(FEATURES):
        raise ValueError("retained features must be declared features")
    new_budget = _nonnegative_finite(new_budget, "new_budget")
    deferred_budget = _nonnegative_finite(deferred_budget, "deferred_budget")
    available = new_budget + deferred_budget
    contributions = _contributions(frame, retained, policy_weights)
    rho_result, scored = _reliability(
        frame,
        rho_values,
        covariance_reference,
        history,
        unscored_rho,
        retained,
        reliability_scale,
    )
    alpha = 1.0 - rho_result
    adjusted = alpha * contributions
    total_adjusted = float(adjusted.sum())
    allocation = allocate_proportional(adjusted, available)
    deferred_out = available if adjusted.size == 0 or np.isclose(total_adjusted, 0.0) else 0.0
    paid = float(allocation.sum())
    budget_error = available - paid - deferred_out
    if not np.isclose(budget_error, 0.0, rtol=0.0, atol=1e-12):
        raise ArithmeticError("CAU budget conservation failed")
    return {
        "contribution": contributions,
        "rho": rho_result,
        "scored": scored,
        "alpha": alpha,
        "adjusted_contribution": adjusted,
        "allocation": allocation,
        "new_budget": new_budget,
        "deferred_budget_in": deferred_budget,
        "available_budget": available,
        "paid_budget": paid,
        "deferred_budget": deferred_out,
        "budget_error": budget_error,
    }


def _cycle_sort(value: Any) -> tuple[int, Any]:
    try:
        number = int(value)
        if float(value) == number:
            return 0, number
    except (TypeError, ValueError, OverflowError):
        pass
    return 1, str(value)


def execute_cycles(
    frame: pd.DataFrame,
    budgets: Mapping[tuple[Any, Any], float],
    *,
    policy_weights: Any = None,
    rho_values: Iterable[float] | None = None,
    covariance_reference: np.ndarray | None = None,
    history: np.ndarray | None = None,
    unscored_rho: float | None = None,
    retained: Iterable[str] = FEATURES,
    reliability_scale: float = 1.5,
) -> dict[str, pd.DataFrame]:
    validate_canonical_keys(frame)
    if not isinstance(budgets, Mapping):
        raise TypeError("budgets must be a mapping keyed by (streamer_id, cycle_id)")
    budget_values: dict[tuple[Any, Any], float] = {}
    for key, value in budgets.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("budget keys must be (streamer_id, cycle_id)")
        budget_values[key] = _nonnegative_finite(value, "new_budget")
    rho_array = None if rho_values is None else np.asarray(list(rho_values), dtype=float)
    if rho_array is not None and rho_array.shape != (len(frame),):
        raise ValueError("rho_values shape mismatch")
    if covariance_reference is not None:
        if rho_array is not None:
            raise ValueError("supply rho_values or covariance_reference, not both")
        rho_array, _ = reliability_from_history(
            frame,
            covariance_reference,
            columns=tuple(retained),
            history=history,
            scale=reliability_scale,
        )
    ordered = frame.assign(_row_position=np.arange(len(frame))).sort_values(
        ["streamer_id", "cycle_id", "user_id"], kind="stable"
    )
    frame_keys = set(frame[list(TENANT_KEYS)].itertuples(index=False, name=None))
    all_keys = frame_keys | set(budget_values)
    tenant_cycles: dict[Any, list[Any]] = {}
    for tenant, cycle in all_keys:
        tenant_cycles.setdefault(tenant, []).append(cycle)
    allocation_rows: list[pd.DataFrame] = []
    cycle_rows: list[dict[str, Any]] = []
    for tenant in sorted(tenant_cycles, key=str):
        deferred = 0.0
        for cycle in sorted(set(tenant_cycles[tenant]), key=_cycle_sort):
            group = ordered[(ordered["streamer_id"] == tenant) & (ordered["cycle_id"] == cycle)]
            if (tenant, cycle) not in budget_values:
                raise KeyError((tenant, cycle))
            positions = group["_row_position"].to_numpy(dtype=int)
            clean_group = group.drop(columns=["_row_position"])
            group_rho = None if rho_array is None else rho_array[positions]
            result = execute_cau(
                clean_group,
                budget_values.get((tenant, cycle), 0.0),
                deferred_budget=deferred,
                policy_weights=policy_weights,
                rho_values=group_rho,
                unscored_rho=unscored_rho,
                retained=retained,
                reliability_scale=reliability_scale,
            )
            deferred = float(result["deferred_budget"])
            cycle_rows.append(
                {
                    "streamer_id": tenant,
                    "cycle_id": cycle,
                    "new_budget": result["new_budget"],
                    "deferred_budget_in": result["deferred_budget_in"],
                    "available_budget": result["available_budget"],
                    "paid_budget": result["paid_budget"],
                    "deferred_budget": result["deferred_budget"],
                    "total_adjusted_contribution": float(result["adjusted_contribution"].sum()),
                    "recipient_count": int(len(clean_group)),
                    "zero_total": bool(np.isclose(result["adjusted_contribution"].sum(), 0.0)),
                    "singleton": bool(len(clean_group) == 1),
                    "budget_error": result["budget_error"],
                }
            )
            if len(clean_group):
                rows = clean_group[list(CANONICAL_KEYS)].copy()
                rows["contribution"] = result["contribution"]
                rows["rho"] = result["rho"]
                rows["alpha"] = result["alpha"]
                rows["adjusted_contribution"] = result["adjusted_contribution"]
                rows["allocation"] = result["allocation"]
                allocation_rows.append(rows)
    allocation_columns = list(CANONICAL_KEYS) + ["contribution", "rho", "alpha", "adjusted_contribution", "allocation"]
    allocations = pd.concat(allocation_rows, ignore_index=True) if allocation_rows else pd.DataFrame(columns=allocation_columns)
    cycles = pd.DataFrame(cycle_rows)
    return {"allocations": allocations, "cycles": cycles}


def proportional_allocate(adjusted_contribution: Iterable[float], budget_available: float) -> tuple[np.ndarray, float]:
    values = np.asarray(list(adjusted_contribution), dtype=float)
    allocation = allocate_proportional(values, budget_available)
    deferred = float(budget_available) if values.size == 0 or np.isclose(values.sum(), 0.0) else 0.0
    return allocation, deferred


def execute_cau_cycles(
    frame: pd.DataFrame,
    covariance_reference: np.ndarray,
    new_budget_column: str = "new_budget",
    history: np.ndarray | None = None,
    scale: float = 1.5,
    unscored_alpha: float | None = None,
    policy_weights: Any = None,
    retained: Iterable[str] = FEATURES,
) -> pd.DataFrame:
    if new_budget_column not in frame.columns:
        raise KeyError(new_budget_column)
    budgets = {}
    for key, group in frame.groupby(list(TENANT_KEYS), sort=False, observed=True):
        values = pd.to_numeric(group[new_budget_column], errors="coerce").to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all() or (values < 0.0).any() or not np.allclose(values, values[0], rtol=0.0, atol=0.0):
            raise ValueError("one finite nonnegative budget is required per tenant-cycle")
        budgets[key] = float(values[0])
    unscored_rho = None if unscored_alpha is None else 1.0 - _nonnegative_finite(unscored_alpha, "unscored_alpha")
    if unscored_rho is not None and not 0.0 <= unscored_rho <= 1.0:
        raise ValueError("unscored_alpha must lie in [0, 1]")
    result = execute_cycles(
        frame,
        budgets,
        policy_weights=policy_weights,
        covariance_reference=covariance_reference,
        history=history,
        unscored_rho=unscored_rho,
        retained=retained,
        reliability_scale=scale,
    )
    allocations = result["allocations"]
    cycles = result["cycles"]
    if allocations.empty:
        return allocations
    return allocations.merge(cycles, on=list(TENANT_KEYS), how="left", validate="many_to_one")


def _validate_integer_inputs(scores: dict[str, int], budget_minor: int) -> None:
    if not isinstance(scores, dict):
        raise TypeError("scores must be a dict")
    if type(budget_minor) is not int:
        raise TypeError("budget_minor must be an integer")
    if budget_minor < 0:
        raise ValueError("budget_minor must be nonnegative")
    for key, value in scores.items():
        if type(key) is not str:
            raise TypeError("recipient IDs must be strings")
        if type(value) is not int:
            raise TypeError("scores must be integers")
        if value < 0:
            raise ValueError("scores must be nonnegative")


def allocate_minor_units(scores: dict[str, int], budget_minor: int) -> dict[str, int]:
    _validate_integer_inputs(scores, budget_minor)
    if not scores:
        if budget_minor:
            raise ValueError("a positive budget requires at least one recipient")
        return {}
    total = sum(scores.values())
    if total == 0:
        return {key: 0 for key in scores}
    floor: dict[str, int] = {}
    remainder: dict[str, int] = {}
    for key, value in scores.items():
        quotient, residual = divmod(budget_minor * value, total)
        floor[key] = quotient
        remainder[key] = residual
    residual_units = budget_minor - sum(floor.values())
    order = sorted(scores, key=lambda key: (-remainder[key], key))
    for key in order[:residual_units]:
        floor[key] += 1
    if sum(floor.values()) != budget_minor:
        raise ArithmeticError("integer allocation failed to conserve budget")
    return floor
