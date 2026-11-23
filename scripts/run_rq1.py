from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stream_policy.attacks.protocols import canonical_attack_name, expected_queue_counts
from stream_policy.detector.metrics import evaluate, select_validation_threshold


MATCHED_BASELINES = {
    "Sybil": "Static-Gini",
    "Rating Stuffing": "matched receiver baseline",
    "Collusion": "Univariate",
    "Cross-Context Injection": "Static-Gini",
}


def _aggregate(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("no evaluation rows")
    counts = {name: int(sum(int(row[name]) for row in rows)) for name in ("tp", "fp", "tn", "fn")}
    tp, fp, tn, fn = (counts[name] for name in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **counts,
        "n_pos": tp + fn,
        "n_neg": tn + fp,
        "n": tp + fp + tn + fn,
        "prediction_rate": (tp + fp) / (tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "seed_count": len(rows),
    }


def _validate_group(group: pd.DataFrame, attack: str, split: str) -> None:
    positive, negative = expected_queue_counts(attack, split)
    if len(group) != positive + negative:
        raise ValueError(f"{attack} {split} denominator mismatch")
    labels = group["label"].to_numpy()
    if not np.isin(labels, [0, 1, False, True]).all():
        raise ValueError("labels must be binary")
    observed = (int(labels.astype(bool).sum()), int((~labels.astype(bool)).sum()))
    if observed != (positive, negative):
        raise ValueError(f"{attack} {split} class counts mismatch")
    scores = pd.to_numeric(group["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("detector scores must be finite")


def _validate_input(frame: pd.DataFrame, expected_seeds: set[int]) -> pd.DataFrame:
    required = {"split", "attack", "label", "score", "baseline_score", "seed"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(missing)
    result = frame.copy()
    result["attack"] = result["attack"].map(canonical_attack_name)
    if set(result["split"].astype(str)) != {"validation", "test"}:
        raise ValueError("RQ1 input must contain validation and test splits only")
    seeds = pd.to_numeric(result["seed"], errors="coerce")
    if seeds.isna().any() or (seeds % 1 != 0).any():
        raise ValueError("seed values must be integers")
    result["seed"] = seeds.astype(int)
    if set(result["seed"]) != expected_seeds:
        raise ValueError("input seeds must exactly match the configured detector seed set")
    baseline = pd.to_numeric(result["baseline_score"], errors="coerce")
    if not np.isfinite(baseline.to_numpy(dtype=float)).all():
        raise ValueError("baseline scores must be finite")
    for attack, attack_group in result.groupby("attack", sort=True):
        for _, seed_group in attack_group.groupby("seed", sort=True):
            for split in ("validation", "test"):
                _validate_group(seed_group[seed_group["split"].eq(split)], attack, split)
    return result


def _run_one(group: pd.DataFrame) -> dict[str, object]:
    validation = group[group["split"].eq("validation")]
    test = group[group["split"].eq("test")]
    threshold = select_validation_threshold(validation["label"].astype(bool), validation["score"])
    result: dict[str, object] = {
        "threshold": threshold,
        "test": evaluate(test["label"].astype(bool), test["score"], threshold),
    }
    baseline = pd.to_numeric(group["baseline_score"], errors="coerce")
    baseline_validation = baseline.loc[validation.index]
    baseline_test = baseline.loc[test.index]
    baseline_threshold = select_validation_threshold(validation["label"].astype(bool), baseline_validation)
    result["matched_baseline_threshold"] = baseline_threshold
    result["matched_baseline_test"] = evaluate(test["label"].astype(bool), baseline_test, baseline_threshold)
    return result


def run(path: Path, seeds_path: Path) -> dict[str, object]:
    seeds_payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    expected_seeds = {int(value) for value in seeds_payload.get("detector_paired", [])}
    if not expected_seeds:
        raise ValueError("seeds file must define detector_paired")
    frame = _validate_input(pd.read_csv(path), expected_seeds)
    result: dict[str, object] = {}
    for attack, group in frame.groupby("attack", sort=True):
        entry: dict[str, object] = {"matched_baseline": MATCHED_BASELINES[attack]}
        if "seed" in frame:
            per_seed: dict[str, object] = {}
            primary_rows = []
            baseline_rows = []
            for seed, seed_group in group.groupby("seed", sort=True):
                current = _run_one(seed_group)
                per_seed[str(int(seed))] = current
                primary_rows.append(current["test"])
                baseline_rows.append(current["matched_baseline_test"])
            entry["per_seed"] = per_seed
            entry["test"] = _aggregate(primary_rows)
            if baseline_rows:
                entry["matched_baseline_test"] = _aggregate(baseline_rows)
        else:
            entry.update(_run_one(group))
        result[attack] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(run(args.input.resolve(strict=True), args.seeds.resolve(strict=True)), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
