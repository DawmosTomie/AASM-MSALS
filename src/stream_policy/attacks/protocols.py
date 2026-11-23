from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AttackSpec:
    name: str
    cohort: str
    positive: int
    negative: int
    allocation_scope: str
    protocol: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.positive + self.negative


ATTACK_SPECS = {
    "Sybil": AttackSpec("Sybil", "actor", 62, 1168, "mapped", ("target 5 percent of clean cohort", "low-account-age and low-device-tier accounts", "zero gifts likes and comments", "clicks uniformly sampled from 50 to 100", "active across five test cycles")),
    "Rating Stuffing": AttackSpec("Rating Stuffing", "receiver", 567, 5095, "detector-only", ("select bottom-allocation-quintile streamers", "set likes and comments to fit-set P99 limits")),
    "Collusion": AttackSpec("Collusion", "actor", 45, 1123, "mapped", ("synchronized groups of 15 accounts", "200 interactions toward one target", "five-minute window")),
    "Cross-Context Injection": AttackSpec("Cross-Context Injection", "actor", 13, 108, "mapped", ("rank-match Shopping inputs one-to-one", "replace with top-10-percent high-allocation Gaming vectors")),
}


VALIDATION_QUEUE_COUNTS = {
    "Sybil": (108, 2035),
    "Rating Stuffing": (1010, 9087),
    "Collusion": (120, 1915),
    "Cross-Context Injection": (26, 225),
}


def canonical_attack_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("attack name must be a string")
    if name == "Cross-Context":
        return "Cross-Context Injection"
    if name not in ATTACK_SPECS:
        raise ValueError(f"unsupported attack: {name}")
    return name


def expected_queue_counts(attack: str, split: str = "test") -> tuple[int, int]:
    attack = canonical_attack_name(attack)
    if split == "test":
        spec = ATTACK_SPECS[attack]
        return spec.positive, spec.negative
    if split == "validation":
        return VALIDATION_QUEUE_COUNTS[attack]
    raise ValueError("split must be validation or test")


def validate_observation_denominator(attack: str, labels: object, split: str = "test") -> None:
    import numpy as np

    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in "biu":
        raise TypeError("labels must be a one-dimensional boolean or integer vector")
    positive, negative = expected_queue_counts(attack, split)
    observed_positive = int(np.count_nonzero(values.astype(bool)))
    observed_negative = int(values.size - observed_positive)
    if (observed_positive, observed_negative) != (positive, negative):
        raise ValueError(
            f"{canonical_attack_name(attack)} {split} denominator mismatch: "
            f"expected {positive}/{negative}, observed {observed_positive}/{observed_negative}"
        )


def validate_queue_contract(queues: Mapping[str, Mapping[str, int]], split: str = "test") -> None:
    if not isinstance(queues, Mapping) or set(queues) != set(ATTACK_SPECS):
        raise ValueError("attack names do not match the locked protocol")
    for name in ATTACK_SPECS:
        row = queues[name]
        if not isinstance(row, Mapping):
            raise TypeError(f"queue row for {name} must be a mapping")
        positive, negative = expected_queue_counts(name, split)
        observed = (row.get("positive", -1), row.get("negative", -1))
        if any(type(value) is not int or value < 0 for value in observed):
            raise TypeError(f"queue counts for {name} must be nonnegative integers")
        if observed != (positive, negative):
            raise ValueError(f"queue denominator mismatch for {name}")


def validate_injection_contract(injection: object, attack: str, split: str = "test", seed: int | None = None) -> None:
    attack = canonical_attack_name(attack)
    labels = getattr(injection, "labels", None)
    manifest = getattr(injection, "manifest", None)
    if labels is None or manifest is None:
        raise TypeError("injection must expose labels and manifest")
    values = list(labels)
    expected_positive, expected_negative = expected_queue_counts(attack, split)
    observed_positive = sum(bool(value) for value in values)
    if (observed_positive, len(values) - observed_positive) != (expected_positive, expected_negative):
        raise ValueError(f"{attack} injection denominator mismatch")
    if len(manifest) != len(values):
        raise ValueError("injection manifest length mismatch")
    if "split" not in manifest or set(manifest["split"].astype(str)) != {split}:
        raise ValueError("injection split mismatch")
    if "attack" not in manifest or set(manifest["attack"].astype(str)) != {attack}:
        raise ValueError("injection manifest attack mismatch")
    if seed is not None and "seed" in manifest and set(manifest["seed"].astype(int)) != {seed}:
        raise ValueError("injection seed mismatch")
