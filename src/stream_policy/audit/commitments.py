from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from rfc8785 import dumps as jcs_dumps


IMPLEMENTATION_DIGEST_PROCEDURE = "r15-implementation-digest-v1"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_DIGEST_TEXT = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


def _digest_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_TEXT.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _hex_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return value.lower()


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = jcs_dumps(value)
    except Exception as exc:
        raise ValueError("value is not RFC 8785 canonicalizable") from exc
    if not isinstance(encoded, bytes):
        raise TypeError("RFC 8785 encoder did not return bytes")
    return encoded


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("sha256 input must be bytes")
    return hashlib.sha256(value).hexdigest()


def commitment(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("implementation file paths must be nonempty strings")
    path = value.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise ValueError("implementation file paths must be relative")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("implementation file paths must be normalized relative paths")
    return "/".join(parts)


def implementation_digest_manifest(
    file_hashes: Mapping[str, str],
    dependencies: Mapping[str, str],
    procedure_version: str = IMPLEMENTATION_DIGEST_PROCEDURE,
) -> dict[str, Any]:
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise ValueError("implementation file hashes are required")
    if not isinstance(dependencies, Mapping):
        raise TypeError("dependency manifest must be a mapping")
    if not isinstance(procedure_version, str) or not procedure_version:
        raise ValueError("implementation digest procedure version is required")
    normalized_files = {_relative_path(path): _hex_digest(digest, f"file hash for {path}") for path, digest in file_hashes.items()}
    if len(normalized_files) != len(file_hashes):
        raise ValueError("implementation file paths must be unique after normalization")
    normalized_dependencies = {}
    for name, version in dependencies.items():
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError("dependency names and versions must be nonempty strings")
        normalized_dependencies[name] = version
    return {
        "procedure_version": procedure_version,
        "files": [{"path": path, "sha256": normalized_files[path]} for path in sorted(normalized_files)],
        "dependencies": {name: normalized_dependencies[name] for name in sorted(normalized_dependencies)},
    }


def implementation_digest(
    file_hashes: Mapping[str, str],
    dependencies: Mapping[str, str],
    procedure_version: str = IMPLEMENTATION_DIGEST_PROCEDURE,
) -> str:
    return commitment(implementation_digest_manifest(file_hashes, dependencies, procedure_version))


def compute_implementation_digest(
    root: str | Path,
    selected_paths: Sequence[str],
    dependencies: Mapping[str, str],
    procedure_version: str = IMPLEMENTATION_DIGEST_PROCEDURE,
) -> str:
    root_path = Path(root).resolve()
    file_hashes: dict[str, str] = {}
    for relative in selected_paths:
        normalized = _relative_path(relative)
        path = (root_path / normalized).resolve()
        try:
            path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError("implementation file path must remain under root") from exc
        if not path.is_file():
            raise ValueError("implementation file path must identify a regular file")
        if normalized in file_hashes:
            raise ValueError("implementation file paths must be unique")
        file_hashes[normalized] = sha256_bytes(path.read_bytes())
    return implementation_digest(file_hashes, dependencies, procedure_version)


def verify_implementation_digest(committed: Any, execution: Any) -> bool:
    try:
        return _hex_digest(committed, "committed implementation digest") == _hex_digest(execution, "execution implementation digest")
    except (TypeError, ValueError):
        return False


def implementation_digest_status(committed: Any, execution: Any) -> str:
    if committed is None or execution is None:
        return "UNAVAILABLE"
    return "MATCH" if verify_implementation_digest(committed, execution) else "MISMATCH"


def canonical_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("canonical record must be a mapping")
    required = {
        "record_kind",
        "tenant_id",
        "cycle_id",
        "context",
        "policy_version",
        "policy",
        "policy_controls",
        "approved_region",
        "authorization_evidence_commitment",
        "ratification_evidence_commitment",
        "predecessor_hash",
        "input_commitment",
        "output_commitment",
        "implementation_digest",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"canonical record is missing fields: {','.join(missing)}")
    if record["record_kind"] != "canonical_record":
        raise ValueError("record_kind must be canonical_record")
    for field in ("tenant_id", "context", "policy_version"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{field} must be a nonempty string")
    if type(record["cycle_id"]) is not int or record["cycle_id"] < 0:
        raise ValueError("cycle_id must be a nonnegative integer")
    if not isinstance(record["policy"], Sequence) or isinstance(record["policy"], (str, bytes)):
        raise ValueError("policy must be a sequence")
    if not isinstance(record["policy_controls"], Mapping):
        raise ValueError("policy_controls must be a mapping")
    _hex_digest(record["predecessor_hash"], "predecessor_hash")
    for field in ("input_commitment", "output_commitment", "implementation_digest"):
        _hex_digest(record[field], field)
    for field in ("authorization_evidence_commitment", "ratification_evidence_commitment"):
        _digest_text(record[field], field)
    if "record_hash" in record or "chain_state" in record:
        raise ValueError("canonical record body cannot contain derived wrapper fields")
    result = dict(record)
    canonical_bytes(result)
    return result


def canonical_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    required = {
        "receipt_kind",
        "receipt_id",
        "candidate_version",
        "predecessor_hash",
        "candidate_commitment",
        "approved_region",
        "eligible_voters",
        "votes",
        "quorum_threshold",
        "acceptance_threshold",
        "tally",
        "activation_condition",
        "authorization_evidence_commitment",
        "ratification_evidence_commitment",
    }
    missing = sorted(required.difference(receipt))
    if missing:
        raise ValueError(f"receipt is missing fields: {','.join(missing)}")
    if receipt["receipt_kind"] not in {"authorization_receipt", "ratification_receipt", "linked_receipt"}:
        raise ValueError("unsupported receipt kind")
    for field in ("receipt_id", "candidate_version", "activation_condition"):
        if not isinstance(receipt[field], str) or not receipt[field]:
            raise ValueError(f"{field} must be a nonempty string")
    _hex_digest(receipt["predecessor_hash"], "predecessor_hash")
    _digest_text(receipt["candidate_commitment"], "candidate_commitment")
    for field in ("authorization_evidence_commitment", "ratification_evidence_commitment"):
        _digest_text(receipt[field], field)
    if not isinstance(receipt["eligible_voters"], Sequence) or isinstance(receipt["eligible_voters"], (str, bytes)):
        raise ValueError("eligible_voters must be a sequence")
    if not isinstance(receipt["votes"], Sequence) or isinstance(receipt["votes"], (str, bytes)):
        raise ValueError("votes must be a sequence")
    result = dict(receipt)
    result["quorum_threshold"] = list(receipt["quorum_threshold"])
    result["acceptance_threshold"] = list(receipt["acceptance_threshold"])
    canonical_bytes(result)
    return result


def canonical_replay_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("replay payload must be a mapping")
    required = {"replay_kind", "scores", "budget_minor_units", "allocation", "deferred_budget", "implementation_digest"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"replay payload is missing fields: {','.join(missing)}")
    if payload["replay_kind"] != "replay_payload":
        raise ValueError("replay_kind must be replay_payload")
    if not isinstance(payload["scores"], Mapping) or not isinstance(payload["allocation"], Mapping):
        raise ValueError("scores and allocation must be mappings")
    for name, values in (("scores", payload["scores"]), ("allocation", payload["allocation"])):
        for key, value in values.items():
            if not isinstance(key, str) or type(value) is not int or value < 0:
                raise ValueError(f"{name} must contain nonnegative integer values keyed by strings")
    for field in ("budget_minor_units", "deferred_budget"):
        if type(payload[field]) is not int or payload[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    _hex_digest(payload["implementation_digest"], "implementation_digest")
    result = dict(payload)
    canonical_bytes(result)
    return result


def chain_state(previous_hash: str, record: Any) -> str:
    return sha256_bytes(bytes.fromhex(_hex_digest(previous_hash, "previous_hash")) + canonical_bytes(record))


def record_hash(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")
    if isinstance(record.get("body"), Mapping):
        return commitment(record["body"])
    return commitment({key: value for key, value in record.items() if key not in {"record_hash", "chain_state"}})


def signed_commitment_envelope(
    chain_state_value: str,
    signature_base64: str,
    algorithm: str = "Ed25519",
    public_key_hex: str | None = None,
) -> dict[str, Any]:
    _hex_digest(chain_state_value, "chain_state")
    if algorithm != "Ed25519":
        raise ValueError("controlled envelope algorithm must be Ed25519")
    if not isinstance(signature_base64, str) or not signature_base64:
        raise ValueError("signature is required")
    envelope = {
        "envelope_kind": "signed_commitment_envelope",
        "algorithm": algorithm,
        "chain_state": chain_state_value.lower(),
        "signature_base64": signature_base64,
    }
    if public_key_hex is not None:
        if not isinstance(public_key_hex, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", public_key_hex):
            raise ValueError("Ed25519 public key must be 32 raw bytes in hexadecimal")
        envelope["public_key_hex"] = public_key_hex.lower()
    canonical_bytes(envelope)
    return envelope


def verify_signed_commitment_envelope(envelope: Mapping[str, Any], public_key_hex: str | None = None) -> bool:
    try:
        if not isinstance(envelope, Mapping) or envelope.get("envelope_kind") != "signed_commitment_envelope":
            return False
        if envelope.get("algorithm") != "Ed25519":
            return False
        key = public_key_hex or envelope.get("public_key_hex")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", key):
            return False
        state = bytes.fromhex(_hex_digest(envelope["chain_state"], "chain_state"))
        signature = base64.b64decode(envelope["signature_base64"], validate=True)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(signature, state)
    except Exception:
        return False
    return True


def verify_ed25519(public_key_hex: str, signature_base64: str, payload: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            base64.b64decode(signature_base64, validate=True), payload
        )
    except Exception:
        return False
    return True


def verify_record_hash(record: dict[str, Any]) -> bool:
    try:
        return record_hash(record) == record.get("record_hash")
    except (TypeError, ValueError):
        return False


def verify_chain_state(record: dict[str, Any]) -> bool:
    try:
        body = record["body"]
        predecessor = body.get("predecessor_hash", body.get("previous_hash"))
        return chain_state(predecessor, body) == record.get("chain_state")
    except (KeyError, TypeError, ValueError):
        return False
