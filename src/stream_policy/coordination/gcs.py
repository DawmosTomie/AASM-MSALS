from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from ..optimization.solver import project_region


REQUIRED_COLUMNS = ("context", "policy_version", "history", "risk", "mean", "gini", "dispersion", "policy")
RANGE_COLUMNS = ("risk", "mean", "gini", "dispersion")


def _validate_summary_rows(
    summaries: pd.DataFrame,
    signature_verifier: Callable[[dict[str, object]], bool] | None,
    expected_policy_version: str,
    expected_schema: str,
    exclude_source_ids: Iterable[object] | None,
    source_id_column: str,
) -> pd.DataFrame:
    if not isinstance(summaries, pd.DataFrame):
        raise TypeError("summaries must be a pandas DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in summaries.columns]
    if missing:
        raise ValueError(f"summary schema is missing columns: {missing}")
    if not isinstance(expected_policy_version, str) or not expected_policy_version or not isinstance(expected_schema, str) or not expected_schema:
        raise ValueError("expected policy version and schema are required")
    if "schema_version" not in summaries.columns:
        raise ValueError("schema_version is required")
    if not summaries["schema_version"].eq(expected_schema).all():
        raise ValueError("summary schema version mismatch")
    if signature_verifier is None or not callable(signature_verifier):
        raise ValueError("an external signature_verifier is required")
    validated = summaries.copy()
    if exclude_source_ids is not None:
        if source_id_column not in validated.columns:
            raise ValueError("source identifier column is required for target exclusion")
        validated = validated.loc[~validated[source_id_column].isin(set(exclude_source_ids))].copy()
    if validated.empty:
        return validated
    valid = pd.Series(True, index=validated.index)
    valid &= validated["context"].map(lambda value: isinstance(value, str) and bool(value)).astype(bool)
    valid &= validated["policy_version"].map(lambda value: isinstance(value, str) and bool(value)).astype(bool)
    valid &= validated["policy_version"].eq(expected_policy_version)
    for column in RANGE_COLUMNS:
        numeric = pd.to_numeric(validated[column], errors="coerce")
        valid &= numeric.notna() & np.isfinite(numeric.to_numpy()) & numeric.between(0.0, 1.0)
        validated[column] = numeric
    history = pd.to_numeric(validated["history"], errors="coerce")
    valid &= history.notna() & np.isfinite(history.to_numpy()) & np.isclose(history.to_numpy(), np.round(history.to_numpy())) & history.ge(0)
    validated["history"] = history
    for index, value in validated["policy"].items():
        try:
            vector = np.asarray(value, dtype=float)
            row_valid = vector.shape == (4,) and np.isfinite(vector).all() and (vector >= 0.0).all() and np.isclose(vector.sum(), 1.0)
        except (TypeError, ValueError):
            row_valid = False
        valid.loc[index] &= row_valid
    for index, row in validated.iterrows():
        try:
            valid.loc[index] &= bool(signature_verifier(row.to_dict()))
        except Exception:
            valid.loc[index] = False
    return validated.loc[valid].copy()


def aggregate_context_summaries(
    summaries: pd.DataFrame,
    context: str,
    min_history: int = 5,
    max_risk: float = 0.15,
    mad_threshold: float = 3.0,
    mad_epsilon: float = 0.0001,
    min_batch: int = 5,
    gamma: float = 2.0,
    *,
    signature_verifier: Callable[[dict[str, object]], bool] | None = None,
    expected_policy_version: str,
    expected_schema: str,
    exclude_source_ids: Iterable[object] | None = None,
    source_id_column: str = "source_id",
    active_channels: int | None = None,
    kappa: float = 4.0 / 3.0,
    source_split_column: str = "split",
) -> tuple[pd.DataFrame, np.ndarray | None]:
    validated = _validate_summary_rows(summaries, signature_verifier, expected_policy_version, expected_schema, exclude_source_ids, source_id_column)
    if source_split_column not in validated.columns:
        raise ValueError("fit-only GCS source selection requires a split column")
    validated = validated.loc[validated[source_split_column].eq("fit")].copy()
    if active_channels is not None:
        if type(active_channels) is not int or active_channels < 0 or not np.isfinite(kappa) or kappa < 0.0:
            raise ValueError("active_channels and kappa must be valid")
        min_batch = max(int(np.ceil(kappa * active_channels)), int(min_batch))
    valid = validated["context"].eq(context) & validated["history"].ge(min_history) & validated["risk"].le(max_risk)
    compatible = validated[valid].copy()
    if len(compatible) < min_batch:
        return compatible, None
    median = float(compatible["mean"].median())
    mad = float(np.median(np.abs(compatible["mean"].to_numpy(dtype=float) - median)))
    denominator = 1.4826 * (mad + mad_epsilon)
    compatible["robust_z"] = np.abs(compatible["mean"] - median) / denominator
    retained = compatible[compatible["robust_z"].le(mad_threshold)].copy()
    if len(retained) < min_batch:
        return retained, None
    means = retained["mean"].to_numpy(dtype=float)
    gini_values = retained["gini"].to_numpy(dtype=float)
    dispersion = retained["dispersion"].to_numpy(dtype=float)
    fitness = means / (means.max() + mad_epsilon) * np.exp(-gamma * (gini_values + dispersion))
    policy = np.vstack(retained["policy"].to_numpy())
    if fitness.sum() <= 0:
        return retained, None
    return retained, np.average(policy, axis=0, weights=fitness)


def warm_start(policy: np.ndarray, current: np.ndarray, radius: float = 0.8) -> tuple[np.ndarray, int]:
    return project_region(np.asarray(policy, dtype=float), np.asarray(current, dtype=float), radius)
