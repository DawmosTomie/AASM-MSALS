from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.features import transform
from .protocols import canonical_attack_name, expected_queue_counts, validate_injection_contract
from .windows import RAW, Windows


ATTACKS = ("Sybil", "Rating Stuffing", "Collusion", "Cross-Context Injection")


@dataclass
class Injection:
    data: Windows
    labels: np.ndarray
    manifest: pd.DataFrame
    synthetic_events: pd.DataFrame
    protocol_note: str


def manifest_for(data: Windows, labels: np.ndarray, attack: str, history: int, group: object = None, donor: object = None, start_ms: int | None = None, seed: int | None = None, donor_context: object = None) -> pd.DataFrame:
    if start_ms is None:
        raise ValueError("start_ms is required")
    required = {"streamer_id", "cycle_id", "entity_key", "context", "split"}
    if not required.issubset(data.metadata.columns):
        raise KeyError(sorted(required.difference(data.metadata.columns)))
    meta = data.metadata[["streamer_id", "cycle_id", "entity_key", "context", "split"]].copy()
    meta.insert(0, "attack", attack)
    meta["positive"] = np.asarray(labels, dtype=bool)
    meta["history_start_cycle"] = meta.cycle_id - history + 1
    meta["score_available_at_utc_ms"] = start_ms + meta.cycle_id.astype("int64") * 86_400_000
    meta["group"] = group if group is not None else ""
    meta["donor_key"] = donor if donor is not None else ""
    meta["donor_context"] = donor_context if donor_context is not None else ""
    if seed is not None:
        meta["seed"] = int(seed)
    meta["n_pos"] = int(np.asarray(labels, dtype=bool).sum())
    meta["n_neg"] = int((~np.asarray(labels, dtype=bool)).sum())
    meta["denominator"] = len(labels)
    meta["prevalence"] = float(np.asarray(labels, dtype=bool).mean()) if len(labels) else np.nan
    return meta


def _sybil_template(fit: dict, count: int, history: int) -> dict[str, np.ndarray]:
    template = fit.get("sybil_raw_template")
    if not isinstance(template, dict):
        raise KeyError("sybil_raw_template")
    result = {}
    for name in RAW:
        if name not in template:
            raise KeyError(f"sybil_raw_template.{name}")
        values = np.asarray(template[name], dtype=float)
        if values.ndim == 1 and values.shape[0] == history:
            values = np.broadcast_to(values, (count, history)).copy()
        if values.shape != (count, history) or not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("sybil raw template must be finite nonnegative count-by-five arrays")
        result[name] = values.copy()
    return result


def sybil(clean: Windows, fit: dict, cfg: dict, rng: np.random.Generator, seed: int, start_ms: int, split: str = "test") -> Injection:
    positive, negative = expected_queue_counts("Sybil", split)
    if len(clean.metadata) != negative or positive <= 0:
        raise ValueError("Sybil requires the locked clean negative cohort")
    history = clean.history
    if int(cfg.get("history_cycles", 5)) != 5 or history != 5:
        raise ValueError("Sybil requires five-cycle windows")
    selected = rng.choice(len(clean.metadata), size=positive, replace=False)
    fake = clean.take(selected)
    template = _sybil_template(fit, positive, history)
    template["click"] = rng.integers(50, 101, size=(positive, history)).astype(float)
    template["gift_amount"][:] = 0.0
    template["like"][:] = 0.0
    template["comment"][:] = 0.0
    fake.metadata["entity_key"] = [f"synthetic:sybil:{seed}:{index}" for index in range(positive)]
    fake.metadata["observed_role"] = "viewer"
    fake.metadata["age_subgroup"] = "new"
    fake.metadata["device_subgroup"] = "low"
    fake.raw = template
    flat = {name: values.reshape(-1) for name, values in template.items()}
    fake.features = transform({name: values for name, values in flat.items()}, fit["feature"] if "feature" in fit else fit).reshape(positive, history, 4)
    data = Windows(pd.concat([clean.metadata, fake.metadata], ignore_index=True), {key: np.concatenate([clean.raw[key], fake.raw[key]]) for key in RAW}, np.concatenate([clean.features, fake.features]))
    labels = np.zeros(len(data.metadata), dtype=bool)
    labels[-positive:] = True
    manifest = manifest_for(data, labels, "Sybil", history, start_ms=start_ms, seed=seed)
    result = Injection(data, labels, manifest, pd.DataFrame(), "Five-cycle synthetic users with externally supplied non-click raw values and zero gifts, likes and comments.")
    validate_injection_contract(result, "Sybil", split, seed)
    return result


def _allocation_column(metadata: pd.DataFrame) -> str:
    for name in ("allocation", "baseline_allocation", "clean_allocation"):
        if name in metadata:
            return name
    raise KeyError("clean baseline allocation column")


def _allocation_values(data: Windows, supplied: object, name: str) -> np.ndarray:
    if supplied is None:
        column = _allocation_column(data.metadata)
        values = pd.to_numeric(data.metadata[column], errors="coerce").to_numpy(dtype=float)
    elif isinstance(supplied, pd.DataFrame):
        if not {"streamer_id", "allocation"}.issubset(supplied.columns):
            raise KeyError("allocation table requires streamer_id and allocation")
        table = supplied[["streamer_id", "allocation"]].copy()
        if table["streamer_id"].duplicated().any():
            raise ValueError("allocation table contains duplicate streamer identifiers")
        values = data.metadata["streamer_id"].map(table.set_index("streamer_id")["allocation"]).to_numpy(dtype=float)
    elif isinstance(supplied, dict):
        values = data.metadata["streamer_id"].map(supplied).to_numpy(dtype=float)
    else:
        values = np.asarray(supplied, dtype=float)
        if values.shape != (len(data.metadata),):
            raise ValueError(f"{name} must contain one value per input row")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    return values


def rating(clean: Windows, fit: dict, caps: dict[str, float], cfg: dict, rng: np.random.Generator, start_ms: int, clean_allocation: object = None, split: str = "test", seed: int | None = None) -> Injection:
    positive, negative = expected_queue_counts("Rating Stuffing", split)
    if len(clean.metadata) != negative:
        raise ValueError("Rating Stuffing requires the locked clean negative cohort")
    allocation = _allocation_values(clean, clean_allocation, "clean_allocation")
    streamer_scores = clean.metadata.assign(_allocation=allocation).groupby("streamer_id", sort=True)["_allocation"].sum()
    quantile = float(cfg.get("rating_allocation_quantile", 0.20))
    if not np.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("rating_allocation_quantile must be in (0, 1]")
    n_streamers = max(1, int(np.ceil(len(streamer_scores) * quantile)))
    low_streamers = set(streamer_scores.sort_values(kind="stable").head(n_streamers).index)
    eligible = np.flatnonzero(clean.metadata["streamer_id"].isin(low_streamers).to_numpy())
    eligible = eligible[(clean.raw["like"][eligible, -1] < float(caps["like"])) | (clean.raw["comment"][eligible, -1] < float(caps["comment"]))]
    if len(eligible) < positive:
        raise ValueError("bottom allocation quintile cannot supply the locked receiver queue")
    chosen = np.sort(rng.choice(eligible, size=positive, replace=False))
    injected = clean.take(chosen)
    injected.metadata["source_entity_key"] = injected.metadata["entity_key"].astype(str)
    injected.metadata["entity_key"] = [f"synthetic:rating:{split}:{index}" for index in range(positive)]
    for key in ("like", "comment"):
        cap = float(caps[key])
        if not np.isfinite(cap) or cap < 0.0:
            raise ValueError("fit-set P99 caps must be finite and nonnegative")
        injected.raw[key][:, :] = cap
    flat = {name: values.reshape(-1) for name, values in injected.raw.items()}
    injected.features = transform({name: values for name, values in flat.items()}, fit["feature"] if "feature" in fit else fit).reshape(positive, injected.history, 4)
    data = Windows(
        pd.concat([clean.metadata, injected.metadata], ignore_index=True),
        {key: np.concatenate([clean.raw[key], injected.raw[key]]) for key in RAW},
        np.concatenate([clean.features, injected.features]),
    )
    labels = np.concatenate([np.zeros(negative, dtype=bool), np.ones(positive, dtype=bool)])
    manifest = manifest_for(data, labels, "Rating Stuffing", data.features.shape[1], start_ms=start_ms, seed=seed)
    manifest["allocation_rank_rule"] = "bottom streamer allocation quintile"
    manifest["streamer_allocation"] = np.concatenate([allocation, allocation[chosen]])
    for key in ("like", "comment"):
        manifest[f"added_{key}"] = np.concatenate([np.zeros(negative), injected.raw[key][:, -1] - clean.raw[key][chosen, -1]])
    result = Injection(data, labels, manifest, pd.DataFrame(), "Receiver-side detector-only queue selected by clean baseline allocation quintile and fit-set P99 caps.")
    validate_injection_contract(result, "Rating Stuffing", split)
    return result


def collusion(clean: Windows, fit: dict, cfg: dict, rng: np.random.Generator, start_ms: int, seed: int | None = None, split: str = "test") -> Injection:
    positive, negative = expected_queue_counts("Collusion", split)
    group_size = int(cfg.get("collusion_group_size", 15))
    event_count = int(cfg.get("collusion_interactions_per_group", cfg.get("collusion_events_per_group", 200)))
    window_seconds = int(cfg.get("collusion_window_seconds", 300))
    if len(clean.metadata) != negative or positive % group_size != 0 or event_count != 200 or group_size != 15 or window_seconds != 300:
        raise ValueError("Collusion does not match the locked queue or group protocol")
    data = clean.take(np.arange(len(clean.metadata)))
    candidates = []
    used_entities: set[object] = set()
    for _, group in data.metadata.groupby(["streamer_id", "cycle_id"], sort=True, observed=True):
        ids = group.index.to_numpy()
        if len(ids) >= group_size:
            order = rng.permutation(ids)
            available = [int(index) for index in order if data.metadata.loc[index, "entity_key"] not in used_entities]
            while len(available) >= group_size:
                bundle = np.asarray(available[:group_size], dtype=int)
                available = available[group_size:]
                entities = set(data.metadata.loc[bundle, "entity_key"])
                if len(entities) != group_size or entities.intersection(used_entities):
                    continue
                used_entities.update(entities)
                candidates.append(bundle)
    group_count = positive // group_size
    if len(candidates) < group_count:
        raise ValueError("insufficient disjoint collusion groups")
    selected = rng.choice(len(candidates), size=group_count, replace=False)
    labels = np.zeros(len(data.metadata), dtype=bool)
    groups = np.full(len(data.metadata), "", dtype=object)
    events = []
    for group_index, candidate_index in enumerate(selected):
        ids = np.asarray(candidates[int(candidate_index)], dtype=int)
        group_id = f"group:{group_index}"
        labels[ids] = True
        groups[ids] = group_id
        members = data.metadata.loc[ids]
        target = members["streamer_id"].iloc[0]
        if members["streamer_id"].nunique() != 1:
            raise ValueError("collusion group must target one streamer")
        start = start_ms + (int(members["cycle_id"].iloc[0]) - 1) * 86_400_000
        for event_index in range(event_count):
            position = ids[event_index % group_size]
            event_type = "like" if event_index % 2 == 0 else "comment"
            data.raw[event_type][position, -1] += 1.0
            events.append({"group": group_id, "target_id": f"streamer:{target}", "target_streamer_id": target, "streamer_id": target, "cycle_id": members["cycle_id"].iloc[0], "entity_key": data.metadata.loc[position, "entity_key"], "event_type": event_type, "timestamp_ms": start + event_index * 1000, "window_start_ms": start, "window_end_ms": start + window_seconds * 1000})
    flat = {name: values[labels].reshape(-1) for name, values in data.raw.items()}
    data.features[labels] = transform({name: values for name, values in flat.items()}, fit["feature"] if "feature" in fit else fit).reshape(int(labels.sum()), data.features.shape[1], 4)
    manifest = manifest_for(data, labels, "Collusion", data.features.shape[1], group=groups, start_ms=start_ms, seed=seed)
    manifest["target_id"] = np.where(labels, manifest["streamer_id"].map(lambda value: f"streamer:{value}"), "")
    result = Injection(data, labels, manifest, pd.DataFrame(events), "Three disjoint groups of 15 accounts with 200 interactions toward one target within five minutes.")
    validate_injection_contract(result, "Collusion", split, seed)
    return result


def cross_context(clean: Windows, fit_gaming: Windows, fit: dict, cfg: dict, rng: np.random.Generator, start_ms: int, seed: int | None = None, gaming_allocation: object = None, shopping_allocation: object = None, split: str = "test") -> Injection:
    positive, negative = expected_queue_counts("Cross-Context Injection", split)
    if len(clean.metadata) != negative or positive > len(clean.metadata):
        raise ValueError("Cross-Context requires the locked Shopping negative cohort")
    if set(fit_gaming.metadata["context"].astype(str).str.lower()) - {"game", "gaming"}:
        raise ValueError("Cross-Context donor pool must be Gaming")
    if set(clean.metadata["context"].astype(str).str.lower()) - {"shop", "shopping"}:
        raise ValueError("Cross-Context target pool must be Shopping")
    donor_alloc = _allocation_values(fit_gaming, gaming_allocation, "gaming_allocation")
    target_alloc = _allocation_values(clean, shopping_allocation, "shopping_allocation")
    streamer_scores = fit_gaming.metadata.assign(_allocation=donor_alloc).groupby("streamer_id", sort=True)["_allocation"].sum()
    donor_top_fraction = float(cfg.get("cross_context_donor_top_fraction", 0.10))
    if not np.isfinite(donor_top_fraction) or not 0 < donor_top_fraction <= 1:
        raise ValueError("cross_context_donor_top_fraction must be in (0, 1]")
    donor_streamers = set(streamer_scores.sort_values(ascending=False, kind="stable").head(max(1, int(np.ceil(len(streamer_scores) * donor_top_fraction)))).index)
    donor_ids = np.flatnonzero(fit_gaming.metadata["streamer_id"].isin(donor_streamers).to_numpy())
    donor_ids = donor_ids[np.argsort(-donor_alloc[donor_ids], kind="stable")]
    if len(donor_ids) < positive:
        raise ValueError("top-decile Gaming donor pool cannot supply the locked queue")
    targets = np.argsort(-target_alloc, kind="stable")[:positive]
    donors = donor_ids[:positive]
    target_positions = np.asarray(targets, dtype=int)
    donor_positions = np.asarray(donors, dtype=int)
    data = clean.take(np.arange(len(clean.metadata)))
    for key in RAW:
        data.raw[key][target_positions] = fit_gaming.raw[key][donor_positions]
    flat = {name: values[target_positions].reshape(-1) for name, values in data.raw.items()}
    data.features[target_positions] = transform({name: values for name, values in flat.items()}, fit["feature"] if "feature" in fit else fit).reshape(positive, data.features.shape[1], 4)
    labels = np.zeros(len(data.metadata), dtype=bool)
    labels[target_positions] = True
    source = fit_gaming.metadata.iloc[donor_positions].reset_index(drop=True)
    donor_keys = np.full(len(labels), "", dtype=object)
    donor_keys[target_positions] = source["entity_key"].astype(str).to_numpy()
    donor_context = np.full(len(labels), "", dtype=object)
    donor_context[target_positions] = source["context"].astype(str).to_numpy()
    manifest = manifest_for(data, labels, "Cross-Context Injection", data.features.shape[1], donor=donor_keys, donor_context=donor_context, start_ms=start_ms, seed=seed)
    donor_streamers = np.full(len(labels), "", dtype=object)
    donor_streamers[target_positions] = source["streamer_id"].astype(str).to_numpy()
    manifest["donor_streamer_id"] = donor_streamers
    manifest["donor_selection"] = np.where(labels, "top-10-percent-Gaming-allocation", "")
    audit = pd.DataFrame({
        "target_streamer_id": data.metadata.iloc[target_positions]["streamer_id"].to_numpy(),
        "target_cycle_id": data.metadata.iloc[target_positions]["cycle_id"].to_numpy(),
        "target_entity_key": data.metadata.iloc[target_positions]["entity_key"].to_numpy(),
        "target_context": data.metadata.iloc[target_positions]["context"].to_numpy(),
        "donor_streamer_id": source["streamer_id"].to_numpy(),
        "donor_cycle_id": source["cycle_id"].to_numpy(),
        "donor_entity_key": source["entity_key"].to_numpy(),
        "donor_context": source["context"].to_numpy(),
        "donor_allocation": donor_alloc[donor_positions],
        "target_allocation": target_alloc[target_positions],
    })
    result = Injection(data, labels, manifest, audit, "Shopping targets are rank-matched one-to-one with top-decile Gaming allocation donors.")
    validate_injection_contract(result, "Cross-Context Injection", split, seed)
    return result


def build(attack: str, clean: Windows, fit: dict, config: dict, seed: int, split: str, start_ms: int, fit_gaming: Windows | None = None, rating_caps: dict[str, float] | None = None, clean_allocation: object = None, gaming_allocation: object = None, shopping_allocation: object = None) -> Injection:
    attack = canonical_attack_name(attack)
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if not len(clean.metadata):
        raise ValueError("empty clean windows")
    salt = 0 if split == "validation" else 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, ATTACKS.index(attack), salt]))
    if attack == "Sybil":
        result = sybil(clean, fit, config, rng, seed, start_ms, split)
    elif attack == "Rating Stuffing":
        if rating_caps is None:
            raise ValueError("rating_caps is required")
        result = rating(clean, fit, rating_caps, config, rng, start_ms, clean_allocation, split, seed)
    elif attack == "Collusion":
        result = collusion(clean, fit, config, rng, start_ms, seed, split)
    else:
        if fit_gaming is None:
            raise ValueError("fit_gaming is required")
        result = cross_context(clean, fit_gaming, fit, config, rng, start_ms, seed, gaming_allocation, shopping_allocation, split)
    validate_injection_contract(result, attack, split, seed)
    return result
