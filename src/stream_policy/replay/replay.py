from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..allocation.cau import allocate_minor_units
from ..audit.commitments import (
    canonical_record,
    canonical_replay_payload,
    chain_state,
    commitment,
    implementation_digest_status,
    record_hash,
    verify_chain_state,
    verify_record_hash,
)


def _validate_integer_budget(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_scores(scores: dict[str, int]) -> None:
    if not isinstance(scores, dict) or any(type(key) is not str or type(value) is not int or value < 0 for key, value in scores.items()):
        raise ValueError("scores must contain nonnegative integer values keyed by strings")


def _zero_allocation(scores: dict[str, int]) -> dict[str, int]:
    return {key: 0 for key in scores}


def replay_allocation(scores: dict[str, int], budget_minor: int) -> dict[str, Any]:
    _validate_integer_budget(budget_minor, "budget_minor")
    _validate_scores(scores)
    if sum(scores.values()) == 0:
        allocation = _zero_allocation(scores)
        deferred_budget = budget_minor
    else:
        allocation = allocate_minor_units(scores, budget_minor)
        deferred_budget = 0
    if sum(allocation.values()) + deferred_budget != budget_minor:
        raise ArithmeticError("replay allocation failed to conserve budget")
    return {"allocation": allocation, "deferred_budget": deferred_budget}


def replay_cycle(scores: dict[str, int], new_budget_minor: int, deferred_budget_in: int = 0) -> dict[str, Any]:
    _validate_integer_budget(new_budget_minor, "new_budget_minor")
    _validate_integer_budget(deferred_budget_in, "deferred_budget_in")
    _validate_scores(scores)
    available_budget = new_budget_minor + deferred_budget_in
    if sum(scores.values()) == 0:
        allocation = _zero_allocation(scores)
        deferred_budget = available_budget
    else:
        allocation = allocate_minor_units(scores, available_budget)
        deferred_budget = 0
    paid_budget = sum(allocation.values())
    if paid_budget + deferred_budget != available_budget:
        raise ArithmeticError("replay cycle failed to conserve budget")
    return {
        "allocation": allocation,
        "new_budget": new_budget_minor,
        "deferred_budget_in": deferred_budget_in,
        "available_budget": available_budget,
        "paid_budget": paid_budget,
        "deferred_budget": deferred_budget,
    }


def build_replay_payload(
    scores: dict[str, int],
    budget_minor: int,
    implementation_digest: str,
) -> dict[str, Any]:
    result = replay_allocation(scores, budget_minor)
    return canonical_replay_payload({
        "replay_kind": "replay_payload",
        "scores": dict(scores),
        "budget_minor_units": budget_minor,
        "allocation": result["allocation"],
        "deferred_budget": result["deferred_budget"],
        "implementation_digest": implementation_digest,
    })


def _cycle_sort(value: Any) -> tuple[int, Any]:
    try:
        number = int(value)
        if float(value) == number:
            return 0, number
    except (TypeError, ValueError, OverflowError):
        pass
    return 1, str(value)


def replay_cycles(cycles: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = list(cycles)
    prepared = []
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("cycle records must be mappings")
        if "tenant_id" not in record or "cycle_id" not in record or "scores" not in record:
            raise KeyError("tenant_id, cycle_id, and scores are required")
        budget_key = "new_budget_minor" if "new_budget_minor" in record else "budget_minor_units"
        if budget_key not in record:
            raise KeyError("new_budget_minor or budget_minor_units is required")
        key = (record["tenant_id"], record["cycle_id"])
        if key in seen:
            raise ValueError("duplicate tenant-cycle replay input")
        seen.add(key)
        prepared.append((str(record["tenant_id"]), record["tenant_id"], record["cycle_id"], record["scores"], record[budget_key]))
    prepared.sort(key=lambda item: (item[0], _cycle_sort(item[2])))
    deferred: dict[Any, int] = {}
    outputs = []
    for _, tenant_id, cycle_id, scores, new_budget in prepared:
        incoming = deferred.get(tenant_id, 0)
        result = replay_cycle(scores, new_budget, incoming)
        deferred[tenant_id] = result["deferred_budget"]
        outputs.append({"tenant_id": tenant_id, "cycle_id": cycle_id, **result})
    return outputs


def _normalise_result(result: dict[str, Any], allow_cycle_fields: bool = False) -> dict[str, Any]:
    if not isinstance(result, dict) or "allocation" not in result or "deferred_budget" not in result:
        raise TypeError("replay result must contain allocation and deferred_budget")
    allowed = {"allocation", "deferred_budget"}
    if allow_cycle_fields:
        allowed |= {"new_budget", "deferred_budget_in", "available_budget", "paid_budget"}
    if set(result).difference(allowed):
        raise TypeError("replay result contains unsupported fields")
    allocation = result["allocation"]
    deferred_budget = result["deferred_budget"]
    if not isinstance(allocation, dict):
        raise TypeError("allocation must be a dict")
    _validate_integer_budget(deferred_budget, "deferred_budget")
    for key, value in allocation.items():
        if type(key) is not str or type(value) is not int or value < 0:
            raise ValueError("allocation must contain nonnegative integer values keyed by strings")
    normalised = {"allocation": dict(allocation), "deferred_budget": deferred_budget}
    for field in ("new_budget", "deferred_budget_in", "available_budget", "paid_budget"):
        if field in result:
            _validate_integer_budget(result[field], field)
            normalised[field] = result[field]
    if allow_cycle_fields and "new_budget" in normalised and "deferred_budget_in" in normalised:
        expected_available = normalised["new_budget"] + normalised["deferred_budget_in"]
        if normalised.get("available_budget", expected_available) != expected_available:
            raise ValueError("replay available budget mismatch")
        if normalised.get("paid_budget", sum(normalised["allocation"].values())) != sum(normalised["allocation"].values()):
            raise ValueError("replay paid budget mismatch")
        if sum(normalised["allocation"].values()) + normalised["deferred_budget"] != expected_available:
            raise ValueError("replay cycle conservation mismatch")
    return normalised


def build_commitment_record(
    tenant_id: str,
    cycle_id: int,
    policy_version: str,
    previous_hash: str,
    inputs: Any,
    outputs: Any,
    *,
    context: str | None = None,
    policy: Any = None,
    policy_controls: Any = None,
    approved_region: Any = None,
    authorization_evidence_commitment: str | None = None,
    ratification_evidence_commitment: str | None = None,
    implementation_digest: str | None = None,
) -> dict[str, Any]:
    if implementation_digest is None:
        raise ValueError("implementation_digest is required for a canonical committed record")
    output_payload = _normalise_result(outputs, allow_cycle_fields=True)
    if isinstance(inputs, Mapping) and inputs.get("replay_kind") == "replay_payload":
        input_payload = canonical_replay_payload(inputs)
        scores = input_payload["scores"]
        budget_minor = input_payload["budget_minor_units"]
    elif isinstance(inputs, Mapping) and "scores" in inputs:
        scores = dict(inputs["scores"])
        budget_minor = inputs.get("budget_minor_units", inputs.get("budget_minor", sum(output_payload["allocation"].values()) + output_payload["deferred_budget"]))
        input_payload = build_replay_payload(scores, budget_minor, implementation_digest)
    else:
        raise TypeError("commitment inputs must contain a replay payload")
    if input_payload["implementation_digest"] != implementation_digest:
        raise ValueError("replay payload and record implementation digests must match")
    if context is None or policy is None or policy_controls is None or approved_region is None:
        raise ValueError("context, policy, controls, and approved region are required")
    if authorization_evidence_commitment is None or ratification_evidence_commitment is None:
        raise ValueError("authorization and ratification evidence commitments are required")
    body = canonical_record({
        "record_kind": "canonical_record",
        "tenant_id": tenant_id,
        "cycle_id": cycle_id,
        "context": context,
        "policy_version": policy_version,
        "policy": policy,
        "policy_controls": policy_controls,
        "approved_region": approved_region,
        "authorization_evidence_commitment": authorization_evidence_commitment,
        "ratification_evidence_commitment": ratification_evidence_commitment,
        "predecessor_hash": previous_hash,
        "input_commitment": commitment(input_payload),
        "output_commitment": commitment({"allocation": output_payload["allocation"], "deferred_budget": output_payload["deferred_budget"]}),
        "implementation_digest": implementation_digest,
    })
    state = chain_state(previous_hash, body)
    envelope = {
        "envelope_kind": "signed_commitment_envelope",
        "algorithm": "Ed25519",
        "chain_state": state,
        "signature_base64": None,
        "status": "EXTERNAL_SIGNATURE_REQUIRED",
    }
    record = {"body": body, "envelope": envelope, "chain_state": state}
    record["record_hash"] = record_hash(record)
    return record


def verify_replay_detailed(
    scores: dict[str, int],
    budget_minor: int,
    result: dict[str, Any],
    *,
    committed_record: dict[str, Any] | None = None,
    execution_implementation_digest: str | None = None,
    replay_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        expected = replay_allocation(scores, budget_minor)
        actual = _normalise_result(result)
    except (TypeError, ValueError, ArithmeticError):
        return {"status": "INVALID_INPUT", "conservation": False}
    if actual != expected:
        return {"status": "REPLAY_MISMATCH", "conservation": False}
    if sum(actual["allocation"].values()) + actual["deferred_budget"] != budget_minor:
        return {"status": "CONSERVATION_MISMATCH", "conservation": False}
    if committed_record is None:
        return {"status": "MATCH", "conservation": True, "implementation_digest": "NOT_CHECKED"}
    body = committed_record.get("body") if isinstance(committed_record, dict) else None
    if not isinstance(body, dict):
        return {"status": "RECORD_MISMATCH", "conservation": True}
    if body.get("implementation_digest") is None or execution_implementation_digest is None:
        return {"status": "IMPLEMENTATION_UNAVAILABLE", "conservation": True, "implementation_digest": "UNAVAILABLE"}
    if not verify_record_hash(committed_record) or not verify_chain_state(committed_record):
        return {"status": "RECORD_MISMATCH", "conservation": True}
    if replay_payload is None:
        try:
            replay_payload = build_replay_payload(scores, budget_minor, execution_implementation_digest)
        except (TypeError, ValueError):
            return {"status": "INPUT_UNAVAILABLE", "conservation": True}
    try:
        replay_payload = canonical_replay_payload(replay_payload)
    except (TypeError, ValueError):
        return {"status": "INPUT_UNAVAILABLE", "conservation": True}
    digest_status = implementation_digest_status(body.get("implementation_digest"), execution_implementation_digest)
    if digest_status != "MATCH":
        return {"status": "IMPLEMENTATION_" + digest_status, "conservation": True, "implementation_digest": digest_status}
    if body.get("input_commitment") != commitment(replay_payload):
        return {"status": "INPUT_MISMATCH", "conservation": True}
    if body.get("output_commitment") != commitment(expected):
        return {"status": "OUTPUT_MISMATCH", "conservation": True}
    return {"status": "MATCH", "conservation": True, "implementation_digest": digest_status}


def verify_replay(
    scores: dict[str, int],
    budget_minor: int,
    result: dict[str, Any],
    *,
    committed_record: dict[str, Any] | None = None,
    execution_implementation_digest: str | None = None,
    replay_payload: dict[str, Any] | None = None,
) -> bool:
    return verify_replay_detailed(
        scores,
        budget_minor,
        result,
        committed_record=committed_record,
        execution_implementation_digest=execution_implementation_digest,
        replay_payload=replay_payload,
    )["status"] == "MATCH"


def verify_replay_cycles(cycles: Iterable[Mapping[str, Any]], results: Iterable[Mapping[str, Any]]) -> bool:
    try:
        expected = replay_cycles(cycles)
        actual = []
        for result in results:
            payload = dict(result)
            tenant_id = payload.pop("tenant_id")
            cycle_id = payload.pop("cycle_id")
            actual.append({"tenant_id": tenant_id, "cycle_id": cycle_id, **_normalise_result(payload, allow_cycle_fields=True)})
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False
    return actual == expected


def replay_payload(
    scores: dict[str, int],
    budget_minor: int,
    result: dict[str, Any],
    implementation_digest: str,
) -> dict[str, Any]:
    normalized = _normalise_result(result)
    return canonical_replay_payload({
        "replay_kind": "replay_payload",
        "scores": dict(scores),
        "budget_minor_units": budget_minor,
        "allocation": normalized["allocation"],
        "deferred_budget": normalized["deferred_budget"],
        "implementation_digest": implementation_digest,
    })


def verify_committed_replay(
    record: dict[str, Any],
    scores: dict[str, int],
    budget_minor: int,
    result: dict[str, Any],
    execution_implementation_digest: str | None,
    inputs: Any = None,
) -> dict[str, Any]:
    detailed = verify_replay_detailed(
        scores,
        budget_minor,
        result,
        committed_record=record,
        execution_implementation_digest=execution_implementation_digest,
        replay_payload=build_replay_payload(scores, budget_minor, record["body"]["implementation_digest"]),
    )
    checks = {
        "record_hash": verify_record_hash(record),
        "chain_state": verify_chain_state(record),
        "implementation_digest": detailed.get("implementation_digest") == "MATCH",
        "input_commitment": detailed.get("status") not in {"INPUT_MISMATCH", "INPUT_UNAVAILABLE"},
        "output_commitment": detailed.get("status") != "OUTPUT_MISMATCH",
        "allocation": detailed.get("status") not in {"REPLAY_MISMATCH", "INVALID_INPUT"},
        "conservation": detailed.get("conservation") is True,
    }
    valid = detailed.get("status") == "MATCH" and all(checks.values())
    return {"valid": valid, "status": "VALID" if valid else detailed.get("status", "INVALID"), "checks": checks}
