from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from stream_policy.optimization.solver import optimize_policy


def _load_config(path: Path) -> dict[str, object]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("optimizer"), dict):
        raise ValueError("config requires an optimizer section")
    optimizer = config["optimizer"]
    heldout = optimizer.get("heldout")
    if not isinstance(heldout, dict):
        raise ValueError("config requires optimizer.heldout settings")
    required = {"eta0", "eta_min", "eta_decay", "finite_difference_step", "allocation_tolerance", "policy_tolerance", "max_iterations", "authorized_l1_radius"}
    missing = sorted(required.difference(heldout))
    if missing:
        raise ValueError(f"held-out optimizer config is missing: {missing}")
    return heldout


def _vector(payload: dict[str, object], name: str, width: int | None = None) -> np.ndarray:
    if name not in payload:
        raise KeyError(name)
    result = np.asarray(payload[name], dtype=float)
    if result.ndim != 1 or (width is not None and result.size != width) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return result


def _matrix(payload: dict[str, object], name: str, width: int) -> np.ndarray:
    if name not in payload:
        raise KeyError(name)
    result = np.asarray(payload[name], dtype=float)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix with {width} columns")
    return result


def run(path: Path, config_path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RQ3 input must be a JSON object")
    current = _vector(payload, "current")
    width = current.size
    train = _matrix(payload, "train", width)
    validation = _matrix(payload, "validation", width)
    risk_train = _vector(payload, "risk_train", train.shape[0])
    risk_validation = _vector(payload, "risk_validation", validation.shape[0])
    arms = {
        "previous_local": current,
        "same_context_mean": _vector(payload, "same_context_mean", width),
        "gcs_weighted": _vector(payload, "gcs_weighted", width),
    }
    settings = _load_config(config_path)
    result = {}
    for name, initial in arms.items():
        rows, summary = optimize_policy(
            initial,
            current,
            train,
            validation,
            risk_train,
            risk_validation,
            eta0=float(settings["eta0"]),
            eta_decay=float(settings["eta_decay"]),
            eta_min=float(settings["eta_min"]),
            max_iterations=int(settings["max_iterations"]),
            allocation_tolerance=float(settings["allocation_tolerance"]),
            policy_tolerance=float(settings["policy_tolerance"]),
            radius=float(settings["authorized_l1_radius"]),
            finite_difference_step=float(settings["finite_difference_step"]),
        )
        result[name] = {"iterations": len(rows), "summary": summary}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run(args.input.resolve(strict=True), args.config.resolve(strict=True)), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
