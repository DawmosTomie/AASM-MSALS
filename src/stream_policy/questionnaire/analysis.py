from __future__ import annotations

import math

import pandas as pd


LIKERT_COLUMNS = ("Q1", "Q2", "Q3", "Q4", "Q5")
PREFERENCE_COLUMNS = ("B1", "B2", "B3", "B4")
PREFERENCE_LABELS = ("A", "B", "Same/No Difference")


def wilson_interval(positive: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if type(positive) is not int or type(total) is not int or total <= 0 or positive < 0 or positive > total:
        raise ValueError("invalid binomial counts")
    p = positive / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _likert_summary(series: pd.Series) -> dict[str, object]:
    missing = _missing_mask(series)
    numeric = pd.to_numeric(series.where(~missing), errors="coerce")
    invalid = (~missing) & (numeric.isna() | ~numeric.between(1, 5))
    if bool(invalid.any()):
        raise ValueError("Likert responses must be integers from 1 to 5")
    valid = numeric.dropna().astype(int)
    positive = int(valid.ge(4).sum())
    total = int(len(valid))
    interval = wilson_interval(positive, total) if total else (None, None)
    return {
        "n": total,
        "valid_n": total,
        "missing": int(missing.sum()),
        "invalid": 0,
        "positive": positive,
        "share": positive / total if total else None,
        "wilson_low": interval[0],
        "wilson_high": interval[1],
    }


def _preference_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in ("Same", "No Difference", "Same/No Difference"):
        return "Same/No Difference"
    if text in ("A", "B"):
        return text
    raise ValueError("preference responses must be A, B, or Same/No Difference")


def _preference_summary(series: pd.Series) -> dict[str, object]:
    values = [_preference_value(value) for value in series.tolist()]
    valid = [value for value in values if value is not None]
    total = len(valid)
    counts = {label: valid.count(label) for label in PREFERENCE_LABELS}
    percentages = {label: counts[label] / total if total else None for label in PREFERENCE_LABELS}
    wilson = {
        label: {
            "positive": counts[label],
            "total": total,
            "share": percentages[label],
            "wilson_low": wilson_interval(counts[label], total)[0] if total else None,
            "wilson_high": wilson_interval(counts[label], total)[1] if total else None,
        }
        for label in PREFERENCE_LABELS
    }
    return {
        "valid_n": total,
        "missing": len(values) - total,
        "invalid": 0,
        "counts": counts,
        "percentages": percentages,
        "wilson": wilson,
    }


def validate_study_design(frame: pd.DataFrame) -> None:
    required = {"participant_id", "role_group", "presentation_order", *LIKERT_COLUMNS, *PREFERENCE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(missing)
    if len(frame) != 45 or frame["participant_id"].isna().any() or frame["participant_id"].nunique() != 45:
        raise ValueError("the declared study has 45 unique participants")
    if frame["role_group"].value_counts().to_dict() != {"Streamer": 15, "Viewer": 15, "Validator": 15}:
        raise ValueError("role groups must contain 15 Streamers, 15 Viewers and 15 Validators")
    if frame["presentation_order"].value_counts().to_dict() != {"A-B": 24, "B-A": 21}:
        raise ValueError("presentation order must contain 24 A-B and 21 B-A records")
    for column in LIKERT_COLUMNS:
        _likert_summary(frame[column])
    for column in PREFERENCE_COLUMNS:
        _preference_summary(frame[column])


def analyze_responses(frame: pd.DataFrame) -> dict[str, object]:
    validate_study_design(frame)
    items = {column: _likert_summary(frame[column]) for column in LIKERT_COLUMNS}
    preferences = {column: _preference_summary(frame[column]) for column in PREFERENCE_COLUMNS}
    return {
        "likert": items,
        "preferences": preferences,
        "B1": preferences["B1"],
        "n": len(frame),
        "roles": frame["role_group"].value_counts().to_dict(),
        "presentation_order": frame["presentation_order"].value_counts().to_dict(),
    }
