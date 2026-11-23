from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import pandas as pd

from stream_policy.allocation.cau import bottom_half_share, execute_cau_cycles, gini
from stream_policy.core.features import FEATURES, transform
from stream_policy.evaluation.allocation import proportional_score_cells, streamer_cluster_bootstrap_difference, summarize_cells, normalized_traffic_score, raw_traffic_score


def _load(path: Path) -> dict:
    if path.suffix.lower() == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_column(frame: pd.DataFrame, name: str) -> str:
    if name in frame:
        return name
    aliases = {"click": ("clicks",), "comment": ("comments",), "like": ("likes",), "gift_amount": ("gift", "gift_amount"), "watch_ms": ("watch",), "age_log": ("account_age",), "device_price_score": ("device_price",)}
    for alias in aliases.get(name, ()):
        if alias in frame:
            return alias
    raise KeyError(name)


def _prepare_features(frame: pd.DataFrame, fit: dict) -> pd.DataFrame:
    result = frame.copy()
    if set(FEATURES).issubset(result.columns):
        values = result[list(FEATURES)].astype(float)
    else:
        feature_fit = fit.get("feature", fit)
        raw = {name: result[_raw_column(result, name)].to_numpy() for name in ("watch_ms", "comment", "like", "gift_amount", "click", "age_log", "device_price_score")}
        values = pd.DataFrame(transform(raw, feature_fit), columns=FEATURES)
    for name in FEATURES:
        result[name] = values[name].to_numpy()
    return result


def _require_input(frame: pd.DataFrame, budget_column: str) -> None:
    required = {"streamer_id", "user_id", "cycle_id", "context", budget_column}
    required.add(_raw_column(frame, "watch_ms"))
    required.add(_raw_column(frame, "click"))
    if not set(FEATURES).issubset(frame.columns):
        required |= {_raw_column(frame, name) for name in ("comment", "like", "gift_amount", "age_log", "device_price_score")}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(missing)
    if frame.duplicated(["streamer_id", "user_id", "cycle_id"]).any():
        raise ValueError("duplicate canonical recipient keys")


def run(path: Path, fit_path: Path, config_path: Path, budget_column: str = "new_budget") -> dict[str, object]:
    frame = pd.read_csv(path)
    _require_input(frame, budget_column)
    fit = _load(fit_path)
    config = _load(config_path)
    traffic_fit = fit["traffic"]
    configured_budget = config["allocation"].get("budget_minor_units")
    if budget_column not in frame:
        raise KeyError(budget_column)
    prepared = _prepare_features(frame, fit)
    clicks = prepared[_raw_column(prepared, "click")]
    watch_ms = prepared["watch_ms"]
    prepared["raw_traffic"] = raw_traffic_score(clicks, watch_ms)
    methods: dict[str, dict[str, object]] = {}
    raw_work, raw_cells = proportional_score_cells(prepared, "raw_traffic", budget_column)
    methods["raw_traffic"] = summarize_cells(raw_cells)
    for click_weight, watch_weight in ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2)):
        name = f"normalized_traffic_{click_weight:.1f}_{watch_weight:.1f}"
        prepared[name] = normalized_traffic_score(clicks, watch_ms, traffic_fit, click_weight, watch_weight)
        _, cells = proportional_score_cells(prepared, name, budget_column)
        methods[name] = summarize_cells(cells)
    reference = fit.get("reference_4d", fit.get("4d"))
    if reference is None:
        raise KeyError("reference_4d or 4d")
    cau_config = config["reliability"]
    cau_work = execute_cau_cycles(prepared, reference, budget_column, scale=float(cau_config["scale"]), unscored_alpha=float(cau_config["unscored_payout_alpha"]))
    cau_cells = []
    for (streamer_id, cycle_id), group in cau_work.groupby(["streamer_id", "cycle_id"], sort=True, observed=True):
        amounts = group["allocation"].to_numpy(dtype=float)
        scores = group["adjusted_contribution"].to_numpy(dtype=float)
        cau_cells.append({"streamer_id": streamer_id, "cycle_id": cycle_id, "n_recipients": len(group), "total_score": float(scores.sum()), "total_paid": float(amounts.sum()), "gini": gini(amounts), "bottom_half_share": bottom_half_share(amounts), "zero_total": bool(scores.sum() == 0.0), "singleton": len(group) == 1, "deferred_budget_in": float(group["deferred_budget_in"].iloc[0]), "deferred_budget": float(group["deferred_budget"].iloc[0]), "available_budget": float(group["available_budget"].iloc[0]), "budget_error": float(group["budget_error"].iloc[0])})
    cau_cells = pd.DataFrame(cau_cells)
    methods["cau"] = summarize_cells(cau_cells)
    paired = {}
    left = raw_cells[["streamer_id", "cycle_id", "gini", "bottom_half_share"]].rename(columns={"gini": "left_gini", "bottom_half_share": "left_bottom"})
    right = cau_cells[["streamer_id", "cycle_id", "gini", "bottom_half_share"]].rename(columns={"gini": "right_gini", "bottom_half_share": "right_bottom"})
    comparison = left.merge(right, on=["streamer_id", "cycle_id"], how="inner").dropna()
    paired["cau_minus_raw_traffic_gini"] = streamer_cluster_bootstrap_difference(comparison, "right_gini", "left_gini", resamples=int(config["bootstrap"]["resamples"]), seed=int(config["bootstrap"]["seed"]))
    paired["cau_minus_raw_traffic_bottom_half"] = streamer_cluster_bootstrap_difference(comparison, "right_bottom", "left_bottom", resamples=int(config["bootstrap"]["resamples"]), seed=int(config["bootstrap"]["seed"]))
    return {"methods": methods, "paired_bootstrap": paired, "configured_budget_minor_units": configured_budget, "historical_results_recomputed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget-column", default="new_budget")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input.resolve(strict=True), args.fit.resolve(strict=True), args.config.resolve(strict=True), args.budget_column)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
