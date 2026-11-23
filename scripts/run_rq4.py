from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stream_policy.allocation.cau import allocate_minor_units


def _allocation(scores: dict[str, int], budget: int) -> np.ndarray:
    result = allocate_minor_units(scores, budget)
    return np.asarray([result[key] for key in sorted(result)], dtype=float)


def _sequence(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite sequence")
    return result.reshape(-1)


def _policy_path(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain at least two finite committed policy vectors")
    if (result < 0.0).any() or not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must contain simplex policy vectors")
    return result


def _variation(path: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(path, axis=0)).sum(axis=1) / 2.0))


def run(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "budget_minor_units" not in payload or "cells" not in payload:
        raise ValueError("RQ4 input requires budget_minor_units and cells")
    budget = payload["budget_minor_units"]
    if type(budget) is not int or budget < 0:
        raise ValueError("budget_minor_units must be a nonnegative integer")
    if not isinstance(payload["cells"], list) or not payload["cells"]:
        raise ValueError("RQ4 cells must be a nonempty list")
    distortions = []
    full_distortions = []
    no_reliability_distortions = []
    full_recovery = []
    ablated_recovery = []
    full_variations = []
    ablated_variations = []
    for cell in payload["cells"]:
        if not isinstance(cell, dict):
            raise TypeError("each RQ4 cell must be an object")
        required = {
            "clean_scores",
            "full_scores",
            "no_reliability_scores",
            "full_recovery_cycles",
            "no_cross_tenant_recovery_cycles",
            "full_committed_policies",
            "no_authorization_committed_policies",
        }
        missing = sorted(required.difference(cell))
        if missing:
            raise KeyError(missing)
        if not isinstance(cell["clean_scores"], dict) or set(cell["clean_scores"]) != set(cell["full_scores"]) or set(cell["full_scores"]) != set(cell["no_reliability_scores"]):
            raise ValueError("matched score dictionaries must have identical recipients")
        clean = _allocation(cell["clean_scores"], budget)
        full = _allocation(cell["full_scores"], budget)
        no_reliability = _allocation(cell["no_reliability_scores"], budget)
        if clean.shape != full.shape or full.shape != no_reliability.shape:
            raise ValueError("matched score dictionaries must have the same recipients")
        full_distortions.append(float(np.abs(full - clean).sum() / (2.0 * budget)) if budget else 0.0)
        no_reliability_distortions.append(float(np.abs(no_reliability - clean).sum() / (2.0 * budget)) if budget else 0.0)
        distortions.append(float(np.abs(full - no_reliability).sum() / (2.0 * budget)) if budget else 0.0)
        full_recovery.extend(_sequence(cell["full_recovery_cycles"], "full_recovery_cycles"))
        ablated_recovery.extend(_sequence(cell["no_cross_tenant_recovery_cycles"], "no_cross_tenant_recovery_cycles"))
        full_variations.append(_variation(_policy_path(cell["full_committed_policies"], "full_committed_policies")))
        ablated_variations.append(_variation(_policy_path(cell["no_authorization_committed_policies"], "no_authorization_committed_policies")))
    return {
        "cells": len(distortions),
        "ablation_metrics": {
            "reliability_adjustment": {
                "full_mean_allocation_distortion": float(np.mean(full_distortions)),
                "no_reliability_mean_allocation_distortion": float(np.mean(no_reliability_distortions)),
                "paired_full_minus_no_reliability": float(np.mean(full_distortions) - np.mean(no_reliability_distortions)),
            },
            "cross_tenant_initialization": {
                "full_mean_recovery_cycles": float(np.mean(full_recovery)),
                "no_cross_tenant_mean_recovery_cycles": float(np.mean(ablated_recovery)),
            },
            "authorization_constraints": {
                "full_mean_committed_weight_variation": float(np.mean(full_variations)),
                "no_authorization_mean_committed_weight_variation": float(np.mean(ablated_variations)),
            },
        },
        "historical_results_recomputed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().write_text(json.dumps(run(args.input.resolve(strict=True)), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
