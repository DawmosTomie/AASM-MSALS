from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..audit.commitments import canonical_bytes, canonical_receipt


class State(str, Enum):
    STABLE = "Stable"
    TRIGGERED = "Triggered"
    AUTHORIZED = "Authorized"
    CANDIDATE = "Candidate"
    VALIDATED = "Validated"
    RATIFIED = "Ratified"
    COMMITTED = "Committed"
    POST_COMMIT = "Post-commit"


STATES = tuple(item.value for item in State)
TRANSITIONS = {
    State.STABLE: State.TRIGGERED,
    State.TRIGGERED: State.AUTHORIZED,
    State.AUTHORIZED: State.CANDIDATE,
    State.CANDIDATE: State.VALIDATED,
    State.VALIDATED: State.RATIFIED,
    State.RATIFIED: State.COMMITTED,
    State.COMMITTED: State.POST_COMMIT,
    State.POST_COMMIT: State.STABLE,
}


def _threshold(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain numerator and denominator")
    numerator, denominator = value
    if type(numerator) is not int or type(denominator) is not int or numerator < 0 or denominator <= 0 or numerator > denominator:
        raise ValueError(f"{name} is invalid")
    return numerator, denominator


def tally_votes(
    eligible: Sequence[dict[str, Any]],
    votes: Sequence[dict[str, Any]],
    quorum: tuple[int, int] = (3, 10),
    acceptance: tuple[int, int] = (3, 4),
) -> dict[str, int | bool]:
    if not isinstance(eligible, Sequence) or isinstance(eligible, (str, bytes)):
        raise TypeError("eligible voters must be a sequence")
    if not isinstance(votes, Sequence) or isinstance(votes, (str, bytes)):
        raise TypeError("votes must be a sequence")
    weights: dict[str, int] = {}
    for entry in eligible:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError("eligible voter IDs must be nonempty strings")
        weight = entry.get("weight")
        if type(weight) is not int or weight < 0 or entry["id"] in weights:
            raise ValueError("eligible voter weights must be unique nonnegative integers")
        weights[entry["id"]] = weight
    identifiers = []
    for vote in votes:
        if not isinstance(vote, dict) or not isinstance(vote.get("id"), str) or vote["id"] not in weights:
            raise ValueError("votes must refer to eligible voters")
        if vote.get("choice") not in ("approve", "reject"):
            raise ValueError("vote choice must be approve or reject")
        identifiers.append(vote["id"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate votes are not allowed")
    qn, qd = _threshold(quorum, "quorum")
    an, ad = _threshold(acceptance, "acceptance")
    total = sum(weights.values())
    cast = sum(weights[identifier] for identifier in identifiers)
    approve = sum(weights[vote["id"]] for vote in votes if vote["choice"] == "approve")
    return {
        "eligible_weight": total,
        "cast_weight": cast,
        "approve_weight": approve,
        "quorum_met": total > 0 and cast > 0 and cast * qd >= total * qn,
        "acceptance_met": total > 0 and cast > 0 and approve * ad >= cast * an,
    }


def verify_controlled_receipt(
    receipt: dict[str, Any],
    *,
    expected_candidate_version: str | None = None,
    expected_predecessor: str | None = None,
    expected_region: Any = None,
    expected_candidate_commitment: str | None = None,
    expected_authorization_commitment: str | None = None,
    expected_ratification_commitment: str | None = None,
    expected_activation_condition: str = "commitment_required",
    expected_receipt_kind: str | None = None,
) -> dict[str, Any]:
    checks = {
        "schema": False,
        "candidate_version": False,
        "predecessor": False,
        "region": False,
        "candidate_commitment": False,
        "tally": False,
        "authorization_commitment": False,
        "ratification_commitment": False,
        "activation_condition": False,
    }
    try:
        normalized = canonical_receipt(receipt)
        checks["schema"] = True
        if expected_receipt_kind is not None and normalized["receipt_kind"] != expected_receipt_kind:
            return {"valid": False, "status": "RECEIPT_KIND_MISMATCH", "checks": checks}
        checks["candidate_version"] = expected_candidate_version is None or normalized["candidate_version"] == expected_candidate_version
        checks["predecessor"] = expected_predecessor is None or normalized["predecessor_hash"] == expected_predecessor
        checks["region"] = expected_region is None or canonical_bytes(normalized["approved_region"]) == canonical_bytes(expected_region)
        checks["candidate_commitment"] = expected_candidate_commitment is None or normalized["candidate_commitment"] == expected_candidate_commitment.lower()
        checks["authorization_commitment"] = expected_authorization_commitment is None or normalized["authorization_evidence_commitment"] == expected_authorization_commitment.lower()
        checks["ratification_commitment"] = expected_ratification_commitment is None or normalized["ratification_evidence_commitment"] == expected_ratification_commitment.lower()
        checks["activation_condition"] = normalized["activation_condition"] == expected_activation_condition
        quorum = _threshold(normalized["quorum_threshold"], "quorum_threshold")
        acceptance = _threshold(normalized["acceptance_threshold"], "acceptance_threshold")
        calculated = tally_votes(normalized["eligible_voters"], normalized["votes"], quorum, acceptance)
        checks["tally"] = normalized["tally"] == calculated
        valid = all(checks.values())
        return {
            "valid": valid,
            "status": "VALID" if valid else "RECEIPT_CHECK_FAILED",
            "checks": checks,
            "tally": calculated,
            "trusted_inputs": ["voter_enrolment", "eligible_set_authority", "threshold_configuration", "key_ownership"],
        }
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return {"valid": False, "status": "RECEIPT_INVALID", "checks": checks, "error": str(exc)}


def ratification_outcome(
    current_policy: Any,
    proposed_policy: Any,
    authorization_passed: bool,
    validation_passed: bool,
    tally: dict[str, int | bool],
) -> dict[str, Any]:
    if not {"quorum_met", "acceptance_met"}.issubset(tally):
        raise ValueError("tally is incomplete")
    ratified = bool(tally["quorum_met"] and tally["acceptance_met"] and authorization_passed and validation_passed)
    return {**tally, "ratified": ratified, "active_policy": proposed_policy if ratified else current_policy}


@dataclass
class GovernanceLifecycle:
    active_version: str = "v0"
    active_policy: Any = None
    state: State = State.STABLE
    candidate_policy: Any = None
    history: list[State] = field(default_factory=lambda: [State.STABLE])

    def transition(self, target: State | str) -> State:
        target = State(target)
        if TRANSITIONS.get(self.state) is not target:
            raise ValueError("invalid lifecycle transition")
        self.state = target
        self.history.append(target)
        return target

    def set_candidate(self, policy: Any) -> Any:
        if self.state is not State.CANDIDATE:
            raise ValueError("candidate policy can only be set in Candidate state")
        self.candidate_policy = policy
        return policy

    def reject(self) -> State:
        self.candidate_policy = None
        self.state = State.STABLE
        self.history.append(State.STABLE)
        return self.state

    def commit(self, version: str, policy: Any = None, commitment_verified: bool = True) -> State:
        if self.state is not State.RATIFIED or not commitment_verified:
            raise ValueError("activation requires ratification and a verified commitment")
        self.active_version = version
        self.active_policy = self.candidate_policy if policy is None else policy
        self.candidate_policy = None
        return self.transition(State.COMMITTED)

    def post_commit(self) -> State:
        return self.transition(State.POST_COMMIT)

    def stabilize(self) -> State:
        if self.state is State.COMMITTED:
            self.post_commit()
        if self.state is not State.POST_COMMIT:
            raise ValueError("post-commit state is required before stabilization")
        return self.transition(State.STABLE)
