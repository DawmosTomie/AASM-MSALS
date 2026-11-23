from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..allocation.cau import allocate_proportional, bottom_half_share, gini


def raw_traffic_score(clicks: Iterable[float], watch_ms: Iterable[float]) -> np.ndarray:
    click_values = np.asarray(list(clicks), dtype=float)
    watch_values = np.asarray(list(watch_ms), dtype=float)
    if click_values.shape != watch_values.shape or not np.isfinite(click_values).all() or not np.isfinite(watch_values).all():
        raise ValueError("invalid traffic vectors")
    if (click_values < 0.0).any() or (watch_values < 0.0).any():
        raise ValueError("traffic values must be nonnegative")
    return 0.5 * click_values + 0.5 * watch_values / 1000.0


def normalized_traffic_score(
    clicks: Iterable[float],
    watch_ms: Iterable[float],
    fit: dict,
    click_weight: float = 0.5,
    watch_weight: float = 0.5,
) -> np.ndarray:
    if not np.isfinite(click_weight) or not np.isfinite(watch_weight) or click_weight < 0.0 or watch_weight < 0.0 or not np.isclose(click_weight + watch_weight, 1.0):
        raise ValueError("traffic weights must be nonnegative and sum to one")
    click_values = np.asarray(list(clicks), dtype=float)
    watch_values = np.asarray(list(watch_ms), dtype=float)
    if click_values.shape != watch_values.shape or not np.isfinite(click_values).all() or not np.isfinite(watch_values).all():
        raise ValueError("invalid traffic vectors")
    if (click_values < 0.0).any() or (watch_values < 0.0).any():
        raise ValueError("traffic values must be nonnegative")
    activity_p99 = float(fit["activity_p99"])
    if not np.isfinite(activity_p99) or activity_p99 < 0.0:
        raise ValueError("invalid fit activity percentile")
    click_values = np.minimum(click_values, activity_p99)
    watch_values = np.log1p(watch_values)
    click_bounds = fit["click"]
    watch_bounds = fit["watch_log"]
    if click_bounds["max"] <= click_bounds["min"] or watch_bounds["max"] <= watch_bounds["min"]:
        raise ValueError("fit normalization bounds must have positive width")
    click_values = np.clip((click_values - float(click_bounds["min"])) / (float(click_bounds["max"]) - float(click_bounds["min"])), 0.0, 1.0)
    watch_values = np.clip((watch_values - float(watch_bounds["min"])) / (float(watch_bounds["max"]) - float(watch_bounds["min"])), 0.0, 1.0)
    return click_weight * click_values + watch_weight * watch_values


def concentration_summary(
    frame: pd.DataFrame,
    amount_column: str = "allocation",
    tenant_columns: tuple[str, str] = ("streamer_id", "cycle_id"),
) -> dict[str, object]:
    required = set(tenant_columns) | {amount_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    rows: list[dict[str, object]] = []
    primary_gini: list[float] = []
    primary_bottom: list[float] = []
    zero_total_cells = 0
    singleton_cells = 0
    for key, group in frame.groupby(list(tenant_columns), sort=True, observed=True):
        amounts = group[amount_column].to_numpy(dtype=float)
        if not np.isfinite(amounts).all() or (amounts < 0.0).any():
            raise ValueError("allocation values must be finite and nonnegative")
        total = float(amounts.sum())
        recipient_count = int(len(amounts))
        zero_total = bool(np.isclose(total, 0.0))
        singleton = bool(recipient_count == 1 and not zero_total)
        if zero_total:
            zero_total_cells += 1
            metric_gini = np.nan
            metric_bottom = np.nan
        elif singleton:
            singleton_cells += 1
            metric_gini = np.nan
            metric_bottom = np.nan
        else:
            metric_gini = gini(amounts)
            metric_bottom = bottom_half_share(amounts)
            primary_gini.append(metric_gini)
            primary_bottom.append(metric_bottom)
        row = {tenant_columns[0]: key[0], tenant_columns[1]: key[1], "recipient_count": recipient_count, "total_paid": total, "zero_total": zero_total, "singleton": singleton, "gini": metric_gini, "bottom_half_share": metric_bottom}
        rows.append(row)
    cell_metrics = pd.DataFrame(rows)
    return {
        "cell_metrics": cell_metrics,
        "primary_cell_count": len(primary_gini),
        "primary_gini_mean": float(np.mean(primary_gini)) if primary_gini else np.nan,
        "primary_bottom_half_share_mean": float(np.mean(primary_bottom)) if primary_bottom else np.nan,
        "zero_total_cells": zero_total_cells,
        "singleton_cells": singleton_cells,
        "excluded_cells": zero_total_cells + singleton_cells,
    }


def proportional_score_cells(
    frame: pd.DataFrame,
    score_column: str,
    budget_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"streamer_id", "cycle_id", score_column, budget_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    if frame.duplicated(["streamer_id", "user_id", "cycle_id"]).any() if "user_id" in frame else False:
        raise ValueError("duplicate allocation rows")
    work = frame.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[budget_column] = pd.to_numeric(work[budget_column], errors="coerce")
    if not np.isfinite(work[score_column].to_numpy(dtype=float)).all() or (work[score_column] < 0.0).any():
        raise ValueError("scores must be finite and nonnegative")
    if not np.isfinite(work[budget_column].to_numpy(dtype=float)).all() or (work[budget_column] < 0.0).any():
        raise ValueError("budgets must be finite and nonnegative")
    allocations = []
    rows = []
    for key, group in work.groupby(["streamer_id", "cycle_id"], sort=True, observed=True):
        budgets = group[budget_column].to_numpy(dtype=float)
        if len(budgets) == 0 or not np.allclose(budgets, budgets[0], rtol=0.0, atol=0.0):
            raise ValueError("one common budget is required per tenant-cycle")
        amounts = allocate_proportional(group[score_column].to_numpy(dtype=float), float(budgets[0]))
        allocation_frame = group[[column for column in ("streamer_id", "user_id", "cycle_id") if column in group]].copy()
        allocation_frame["allocation"] = amounts
        allocations.append(allocation_frame)
        total_score = float(group[score_column].sum())
        rows.append({
            "streamer_id": key[0],
            "cycle_id": key[1],
            "recipient_count": int(len(group)),
            "total_paid": float(amounts.sum()),
            "zero_total": bool(np.isclose(total_score, 0.0)),
            "singleton": bool(len(group) == 1 and not np.isclose(total_score, 0.0)),
            "gini": gini(amounts) if total_score > 0.0 and len(group) > 1 else np.nan,
            "bottom_half_share": bottom_half_share(amounts) if total_score > 0.0 and len(group) > 1 else np.nan,
            "budget_error": float(budgets[0] - amounts.sum()) if total_score > 0.0 else float(budgets[0]),
        })
    allocation_rows = pd.concat(allocations, ignore_index=True) if allocations else pd.DataFrame(columns=["streamer_id", "user_id", "cycle_id", "allocation"])
    return allocation_rows, pd.DataFrame(rows)


def summarize_cells(cells: pd.DataFrame) -> dict[str, object]:
    required = {"gini", "bottom_half_share", "zero_total", "singleton"}
    missing = required.difference(cells.columns)
    if missing:
        raise KeyError(sorted(missing))
    primary = cells.loc[~cells["zero_total"].astype(bool) & ~cells["singleton"].astype(bool)]
    return {
        "cell_count": int(len(cells)),
        "primary_cell_count": int(len(primary)),
        "primary_gini_mean": float(primary["gini"].mean()) if len(primary) else None,
        "primary_bottom_half_share_mean": float(primary["bottom_half_share"].mean()) if len(primary) else None,
        "zero_total_cells": int(cells["zero_total"].astype(bool).sum()),
        "singleton_cells": int(cells["singleton"].astype(bool).sum()),
        "excluded_cells": int((cells["zero_total"].astype(bool) | cells["singleton"].astype(bool)).sum()),
    }


def streamer_cluster_bootstrap_difference(
    frame: pd.DataFrame,
    right_column: str,
    left_column: str,
    resamples: int = 2000,
    seed: int = 0,
    streamer_col: str = "streamer_id",
) -> dict[str, object]:
    required = {streamer_col, right_column, left_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(sorted(missing))
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    values = pd.DataFrame({
        streamer_col: frame[streamer_col],
        "difference": pd.to_numeric(frame[right_column], errors="coerce") - pd.to_numeric(frame[left_column], errors="coerce"),
    }).dropna()
    grouped = values.groupby(streamer_col, observed=True)["difference"].mean()
    if grouped.empty:
        return {"estimate": None, "ci_low": None, "ci_high": None, "clusters": 0, "resamples": resamples}
    data = grouped.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    return {
        "estimate": float(data.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(data)),
        "resamples": resamples,
    }


def streamer_bootstrap(
    frame: pd.DataFrame,
    value_columns: tuple[str, ...],
    streamer_col: str = "streamer_id",
    resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    if type(resamples) is not int or resamples <= 0 or streamer_col not in frame or any(column not in frame for column in value_columns):
        raise ValueError("invalid bootstrap input")
    grouped = frame.groupby(streamer_col, observed=True)[list(value_columns)].mean().dropna()
    if grouped.empty:
        return pd.DataFrame(columns=["metric", "estimate", "ci_low", "ci_high", "clusters"])
    rng = np.random.default_rng(seed)
    values = grouped.to_numpy(dtype=float)
    rows = []
    for index, column in enumerate(value_columns):
        draws = rng.choice(values[:, index], size=(resamples, len(values)), replace=True).mean(axis=1)
        rows.append({"metric": column, "estimate": float(values[:, index].mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "clusters": int(len(values))})
    return pd.DataFrame(rows)
