from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from stream_policy.audit.commitments import commitment, compute_implementation_digest, implementation_digest, implementation_digest_status, verify_implementation_digest
from stream_policy.governance.lifecycle import GovernanceLifecycle, State, verify_controlled_receipt
from stream_policy.replay.replay import build_commitment_record, replay_allocation, verify_committed_replay


def _config_values(config: dict[str, Any]) -> dict[str, Any]:
    governance = config.get("governance")
    contribution = config.get("contribution")
    if not isinstance(governance, dict) or not isinstance(contribution, dict):
        raise ValueError("config requires governance and contribution sections")
    required = {
        "quorum_numerator",
        "quorum_denominator",
        "acceptance_numerator",
        "acceptance_denominator",
        "voting_units",
        "cast_units",
        "approve_units",
    }
    if not required.issubset(governance) or not isinstance(contribution.get("features"), list) or not contribution["features"]:
        raise ValueError("config is incomplete for the controlled benchmark")
    return {
        "quorum": (governance["quorum_numerator"], governance["quorum_denominator"]),
        "acceptance": (governance["acceptance_numerator"], governance["acceptance_denominator"]),
        "voting_units": governance["voting_units"],
        "cast_units": governance["cast_units"],
        "approve_units": governance["approve_units"],
        "feature_count": len(contribution["features"]),
    }


def _require_payload(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "scores",
        "budget_minor_units",
        "tenant_id",
        "cycle_id",
        "context",
        "policy_version",
        "policy",
        "policy_controls",
        "approved_region",
        "previous_hash",
        "authorization_receipt",
        "ratification_receipt",
        "implementation",
        "committed_implementation_digest",
        "execution_implementation_digest",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"benchmark input is missing: {','.join(missing)}")
    if type(payload["budget_minor_units"]) is not int or payload["budget_minor_units"] < 0:
        raise ValueError("budget_minor_units must be a nonnegative integer")
    if not isinstance(payload["scores"], dict) or not payload["scores"]:
        raise ValueError("scores must be a nonempty mapping")
    if not isinstance(payload["tenant_id"], str) or not isinstance(payload["context"], str) or not isinstance(payload["policy_version"], str):
        raise ValueError("tenant_id, context and policy_version must be strings")
    if type(payload["cycle_id"]) is not int or payload["cycle_id"] < 0:
        raise ValueError("cycle_id must be a nonnegative integer")
    if not isinstance(payload["policy"], list) or len(payload["policy"]) != config["feature_count"]:
        raise ValueError("policy dimension does not match config")
    if not isinstance(payload["policy_controls"], dict) or not isinstance(payload["implementation"], dict):
        raise ValueError("policy_controls and implementation must be mappings")
    implementation = payload["implementation"]
    if not isinstance(implementation.get("files"), dict) or not isinstance(implementation.get("dependencies"), dict):
        raise ValueError("implementation requires file hashes and dependency versions")
    return payload


def _implementation_values(payload: dict[str, Any], implementation_root: Path | None = None, selected_paths: list[str] | None = None) -> tuple[str, str, str]:
    implementation = payload["implementation"]
    manifest_digest = implementation_digest(
        implementation["files"],
        implementation["dependencies"],
        implementation.get("procedure_version", "r15-implementation-digest-v1"),
    )
    committed = payload["committed_implementation_digest"]
    declared_execution = payload["execution_implementation_digest"]
    if not isinstance(committed, str) or not isinstance(declared_execution, str):
        raise ValueError("committed and execution implementation digests must be strings")
    if not verify_implementation_digest(declared_execution, manifest_digest):
        raise ValueError("execution digest does not match the supplied implementation manifest")
    if implementation_root is None:
        return committed.lower(), declared_execution.lower(), "external_manifest"
    paths = selected_paths or sorted(implementation["files"])
    execution = compute_implementation_digest(
        implementation_root,
        paths,
        implementation["dependencies"],
        implementation.get("procedure_version", "r15-implementation-digest-v1"),
    )
    return committed.lower(), execution, "computed_from_files"


def _candidate(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    candidate = {
        "candidate_kind": "policy_candidate",
        "context": payload["context"],
        "policy_version": payload["policy_version"],
        "policy": list(payload["policy"]),
        "policy_controls": dict(payload["policy_controls"]),
        "approved_region": payload["approved_region"],
    }
    return candidate, commitment(candidate)


def _validate_candidate(candidate: dict[str, Any]) -> None:
    policy = candidate["policy"]
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0.0 for value in policy):
        raise ValueError("candidate policy contains an invalid weight")
    if not math.isclose(sum(float(value) for value in policy), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("candidate policy must sum to one")


def _receipt_check(
    receipt: dict[str, Any],
    values: dict[str, Any],
    predecessor: str,
    region: Any,
    candidate_commitment: str | None = None,
    expected_receipt_kind: str | None = None,
) -> dict[str, Any]:
    result = verify_controlled_receipt(
        receipt,
        expected_candidate_version=values["policy_version"],
        expected_predecessor=predecessor,
        expected_region=region,
        expected_candidate_commitment=candidate_commitment,
        expected_activation_condition="commitment_required",
        expected_authorization_commitment=receipt["authorization_evidence_commitment"],
        expected_ratification_commitment=receipt["ratification_evidence_commitment"],
        expected_receipt_kind=expected_receipt_kind,
    )
    if not result["valid"]:
        raise ValueError(f"controlled receipt check failed: {result['status']}")
    tally = result["tally"]
    if (
        tally["eligible_weight"] != values["voting_units"]
        or tally["cast_weight"] != values["cast_units"]
        or tally["approve_weight"] != values["approve_units"]
    ):
        raise ValueError("receipt does not match the configured voting-unit fixture")
    return result


def _measure(timings: dict[str, float], name: str, action: Callable[[], Any]) -> Any:
    started = time.perf_counter_ns()
    result = action()
    timings[name] += (time.perf_counter_ns() - started) / 1_000_000.0
    return result


def run(payload: dict[str, Any], runs: int, config: dict[str, Any], implementation_root: Path | None = None, selected_paths: list[str] | None = None) -> dict[str, Any]:
    if type(runs) is not int or runs <= 0:
        raise ValueError("runs must be positive")
    values = _config_values(config)
    payload = _require_payload(payload, values)
    committed_digest, execution_digest, digest_source = _implementation_values(payload, implementation_root, selected_paths)
    timings = {
        "core_allocation": 0.0,
        "evidence_aggregation": 0.0,
        "authorization": 0.0,
        "candidate_generation": 0.0,
        "candidate_validation": 0.0,
        "ratification": 0.0,
        "commitment": 0.0,
        "deterministic_replay": 0.0,
    }
    previous = payload["previous_hash"]
    final_status = None
    for offset in range(runs):
        cycle_id = payload["cycle_id"] + offset
        authorization_receipt = deepcopy(payload["authorization_receipt"])
        ratification_receipt = deepcopy(payload["ratification_receipt"])
        authorization_receipt["predecessor_hash"] = previous
        ratification_receipt["predecessor_hash"] = previous
        allocation_result = _measure(
            timings,
            "core_allocation",
            lambda: replay_allocation(payload["scores"], payload["budget_minor_units"]),
        )
        _measure(
            timings,
            "evidence_aggregation",
            lambda: commitment({"authorization": authorization_receipt, "ratification": ratification_receipt}),
        )
        _measure(
            timings,
            "authorization",
            lambda: _receipt_check(authorization_receipt, values | {"policy_version": payload["policy_version"]}, previous, payload["approved_region"], expected_receipt_kind="authorization_receipt"),
        )
        candidate, candidate_hash = _measure(timings, "candidate_generation", lambda: _candidate(payload))
        if authorization_receipt["candidate_commitment"].lower() != candidate_hash:
            raise ValueError("authorization receipt candidate commitment mismatch")
        if ratification_receipt["candidate_commitment"].lower() != candidate_hash:
            raise ValueError("ratification receipt candidate commitment mismatch")
        _measure(timings, "candidate_validation", lambda: _validate_candidate(candidate))
        _measure(
            timings,
            "ratification",
            lambda: _receipt_check(ratification_receipt, values | {"policy_version": payload["policy_version"]}, previous, payload["approved_region"], candidate_hash, "ratification_receipt"),
        )
        lifecycle = GovernanceLifecycle(active_version=payload["policy_version"])
        lifecycle.transition(State.TRIGGERED)
        lifecycle.transition(State.AUTHORIZED)
        lifecycle.transition(State.CANDIDATE)
        lifecycle.set_candidate(candidate["policy"])
        lifecycle.transition(State.VALIDATED)
        lifecycle.transition(State.RATIFIED)
        lifecycle.commit(payload["policy_version"], commitment_verified=True)
        lifecycle.post_commit()
        lifecycle.stabilize()
        record = _measure(
            timings,
            "commitment",
            lambda: build_commitment_record(
                payload["tenant_id"],
                cycle_id,
                payload["policy_version"],
                previous,
                {"scores": payload["scores"], "budget_minor_units": payload["budget_minor_units"], "cycle_id": cycle_id},
                allocation_result,
                context=payload["context"],
                policy=payload["policy"],
                policy_controls=payload["policy_controls"],
                approved_region=payload["approved_region"],
                authorization_evidence_commitment=commitment(authorization_receipt),
                ratification_evidence_commitment=commitment(ratification_receipt),
                implementation_digest=committed_digest,
            ),
        )
        replay_status = _measure(
            timings,
            "deterministic_replay",
            lambda: verify_committed_replay(
                record,
                payload["scores"],
                payload["budget_minor_units"],
                allocation_result,
                execution_digest,
                {"scores": payload["scores"], "budget_minor_units": payload["budget_minor_units"], "cycle_id": cycle_id},
            ),
        )
        if not replay_status["valid"]:
            raise ValueError(f"replay verification failed: {replay_status['status']}")
        final_status = replay_status["status"]
        previous = record["chain_state"]
    stage_results = {
        name: {"runs": runs, "total_ms": total, "mean_ms": total / runs} for name, total in timings.items()
    }
    total_ms = sum(timings.values())
    return {
        "evidence_class": "controlled_local_benchmark",
        "runs": runs,
        "configured_voting_units": {"eligible": values["voting_units"], "cast": values["cast_units"], "approve": values["approve_units"]},
        "stages": stage_results,
        "total": {"total_ms": total_ms, "mean_ms": total_ms / runs},
        "replay_status": final_status,
        "implementation_digest_status": implementation_digest_status(committed_digest, execution_digest),
        "implementation_digest_source": digest_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--implementation-root", type=Path)
    parser.add_argument("--implementation-file", action="append", dest="implementation_files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.resolve(strict=True).read_text(encoding="utf-8"))
    payload = json.loads(args.input.resolve(strict=True).read_text(encoding="utf-8"))
    result = run(payload, args.runs, config, args.implementation_root.resolve() if args.implementation_root else None, args.implementation_files)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
