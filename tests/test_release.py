import base64
import unittest

import numpy as np
import pandas as pd

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stream_policy.audit.commitments import canonical_bytes, chain_state, commitment, verify_ed25519
from stream_policy.attacks.injections import sybil
from stream_policy.attacks.protocols import ATTACK_SPECS, validate_queue_contract
from stream_policy.attacks.windows import RAW, Windows
from stream_policy.coordination.gcs import aggregate_context_summaries
from stream_policy.core.features import reliability_from_history, transform, weights_for_context
from stream_policy.detector.metrics import evaluate, select_validation_threshold
from stream_policy.governance.lifecycle import GovernanceLifecycle, State, tally_votes
from stream_policy.preprocessing.ledger import assign_locked_split, canonicalize_events, validate_canonical_rows
from stream_policy.questionnaire.analysis import wilson_interval
from stream_policy.replay.replay import build_commitment_record, replay_allocation, replay_cycles, verify_replay, verify_replay_cycles
from stream_policy.simulation.transition import configured_transition
from stream_policy.allocation.cau import allocate_minor_units, allocate_proportional, bottom_half_share, execute_cau, execute_cycles
from stream_policy.evaluation.allocation import concentration_summary, normalized_traffic_score, raw_traffic_score
from stream_policy.optimization.solver import project_region


class ReleaseTests(unittest.TestCase):
    def test_jcs_hash_and_signature(self):
        value = {"é": "值", "n": 1.0, "a": [True, None]}
        self.assertEqual(canonical_bytes({"b": 1, "a": 2}), b'{"a":2,"b":1}')
        self.assertEqual(commitment(value), commitment({"a": [True, None], "n": 1.0, "é": "值"}))
        with self.assertRaises(ValueError):
            canonical_bytes(float("nan"))
        key = Ed25519PrivateKey.generate()
        payload = canonical_bytes(value)
        signature = key.sign(payload)
        public = key.public_key().public_bytes_raw().hex()
        self.assertTrue(verify_ed25519(public, base64.b64encode(signature).decode("ascii"), payload))
        self.assertEqual(chain_state("00" * 32, value), chain_state("00" * 32, value))

    def test_integer_allocation_and_deferred_replay(self):
        self.assertEqual(allocate_minor_units({"b": 1, "a": 1}, 1), {"b": 0, "a": 1})
        self.assertEqual(allocate_minor_units({"a": 0, "b": 0}, 7), {"a": 0, "b": 0})
        with self.assertRaises((TypeError, ValueError)):
            allocate_minor_units({"a": 1.0}, 7)
        result = replay_allocation({"a": 0, "b": 0}, 100)
        self.assertEqual(result, {"allocation": {"a": 0, "b": 0}, "deferred_budget": 100})
        self.assertTrue(verify_replay({"a": 0, "b": 0}, 100, result))
        result = replay_allocation({"a": 2, "b": 1}, 10)
        self.assertEqual(sum(result["allocation"].values()) + result["deferred_budget"], 10)
        record = build_commitment_record(
            "tenant",
            1,
            "v1",
            "00" * 32,
            {"scores": {"a": 2, "b": 1}, "budget_minor_units": 10},
            result,
            context="game",
            policy=[0.6, 0.19, 0.15, 0.06],
            policy_controls={"reliability_scale": 1.5},
            approved_region={"radius": 0.8},
            authorization_evidence_commitment="11" * 32,
            ratification_evidence_commitment="22" * 32,
            implementation_digest="00" * 32,
        )
        self.assertEqual(record["body"]["output_commitment"], commitment({"allocation": result["allocation"], "deferred_budget": 0}))

    def test_cau_continuous_path_and_carry_forward(self):
        frame = pd.DataFrame([
            {"streamer_id": "s", "user_id": "u1", "cycle_id": 1, "context": "game", "engagement": 0.0, "financial": 0.0, "activity": 0.0, "authenticity": 0.0},
            {"streamer_id": "s", "user_id": "u2", "cycle_id": 1, "context": "game", "engagement": 0.0, "financial": 0.0, "activity": 0.0, "authenticity": 0.0},
            {"streamer_id": "s", "user_id": "u1", "cycle_id": 2, "context": "game", "engagement": 1.0, "financial": 0.0, "activity": 0.0, "authenticity": 0.0},
            {"streamer_id": "s", "user_id": "u2", "cycle_id": 2, "context": "game", "engagement": 0.0, "financial": 1.0, "activity": 0.0, "authenticity": 0.0},
        ])
        first = execute_cau(frame.iloc[:2], 100.0, rho_values=[0.0, 0.0])
        self.assertEqual(first["deferred_budget"], 100.0)
        self.assertEqual(first["paid_budget"], 0.0)
        attenuated = execute_cau(frame.iloc[2:], 10.0, rho_values=[1.0, 0.0])
        self.assertAlmostEqual(float(attenuated["alpha"][0]), 0.0)
        self.assertAlmostEqual(float(attenuated["allocation"][0]), 0.0)
        self.assertAlmostEqual(float(attenuated["allocation"].sum()), 10.0)
        result = execute_cycles(frame, {("s", 1): 100.0, ("s", 2): 50.0}, rho_values=[0.0] * 4)
        self.assertEqual(result["cycles"]["available_budget"].tolist(), [100.0, 150.0])
        self.assertEqual(result["cycles"]["deferred_budget"].tolist(), [100.0, 0.0])
        self.assertEqual(result["cycles"]["paid_budget"].tolist(), [0.0, 150.0])
        self.assertAlmostEqual(float(result["allocations"]["allocation"].sum()), 150.0)
        with self.assertRaises(ValueError):
            execute_cau(pd.concat([frame.iloc[:1], frame.iloc[:1]]), 10.0, rho_values=[0.0, 0.0])
        replayed = replay_cycles([
            {"tenant_id": "s", "cycle_id": 1, "scores": {"u1": 0, "u2": 0}, "new_budget_minor": 100},
            {"tenant_id": "s", "cycle_id": 2, "scores": {"u1": 2, "u2": 1}, "new_budget_minor": 50},
        ])
        self.assertEqual(replayed[1]["available_budget"], 150)
        self.assertEqual(sum(replayed[1]["allocation"].values()) + replayed[1]["deferred_budget"], 150)
        self.assertTrue(verify_replay_cycles([
            {"tenant_id": "s", "cycle_id": 1, "scores": {"u1": 0, "u2": 0}, "new_budget_minor": 100},
            {"tenant_id": "s", "cycle_id": 2, "scores": {"u1": 2, "u2": 1}, "new_budget_minor": 50},
        ], replayed))
        fractional = allocate_proportional([1.0, 2.0, 3.0], 10.0)
        self.assertTrue(np.allclose(fractional, [10.0 / 6.0, 20.0 / 6.0, 5.0]))
        self.assertTrue(np.allclose(raw_traffic_score([10.0], [1000.0]), [5.5]))
        self.assertTrue(np.allclose(normalized_traffic_score([10.0], [1000.0], {"activity_p99": 10.0, "click": {"min": 0.0, "max": 10.0}, "watch_log": {"min": 0.0, "max": np.log1p(1000.0)}}), [1.0]))
        history = pd.DataFrame([
            {"streamer_id": "s", "user_id": "u", "cycle_id": cycle, "engagement": 0.5, "financial": 0.5, "activity": 0.5, "authenticity": 0.5}
            for cycle in range(1, 7)
        ])
        history_scores, scored = reliability_from_history(history, np.zeros((4, 4)))
        self.assertEqual(scored.tolist(), [False, False, False, False, False, True])
        self.assertTrue(np.isnan(history_scores[:5]).all())
        self.assertAlmostEqual(float(history_scores[5]), 0.0)

    def test_concentration_excludes_zero_and_singleton_cells(self):
        cells = pd.DataFrame({"streamer_id": ["s", "s", "t", "u"], "cycle_id": [1, 2, 1, 1], "allocation": [0.0, 10.0, 5.0, 0.0]})
        summary = concentration_summary(cells)
        self.assertEqual(summary["primary_cell_count"], 0)
        self.assertEqual(summary["zero_total_cells"], 2)
        self.assertEqual(summary["singleton_cells"], 2)
        self.assertEqual(summary["excluded_cells"], 4)

    def test_features_preprocessing_metrics_and_transition(self):
        self.assertEqual(assign_locked_split([1, 12, 13, 16, 17, 21]).tolist(), ["fit", "fit", "validation", "validation", "test", "test"])
        events = pd.DataFrame([
            {"streamer_id": "s", "user_id": "u", "timestamp": "2026-01-01T00:00:00Z", "click": 1, "comment": 0, "like": 0, "gift": 0},
            {"streamer_id": "s", "user_id": "u", "timestamp": "2026-01-01T01:00:00Z", "click": 2, "comment": 0, "like": 0, "gift": 0},
            {"streamer_id": "s", "user_id": "s", "timestamp": "2026-01-01T01:00:00Z", "click": 2, "comment": 0, "like": 0, "gift": 0},
        ])
        rows = canonicalize_events(events, "2026-01-01T00:00:00+08:00")
        validate_canonical_rows(rows)
        self.assertEqual(int(rows.iloc[0].click), 3)
        raw = {"watch_ms": np.array([0.0, 1000.0]), "comment": np.array([0.0, 1.0]), "like": np.array([0.0, 1.0]), "gift_amount": np.array([0.0, 1.0]), "click": np.array([0.0, 1.0]), "age_log": np.array([0.0, 1.0]), "device_price_score": np.array([0.0, 1.0])}
        unit = {"min": 0.0, "max": 1.0, "missing_median": 0.5}
        result = transform(raw, {"columns": {name: unit for name in ("watch_log", "comment", "like", "gift_amount", "activity_capped", "age_log", "device_price_score")}, "activity_p99": 2.0})
        self.assertEqual(result.shape, (2, 4))
        self.assertAlmostEqual(float(weights_for_context("game").sum()), 1.0)
        threshold = select_validation_threshold([False, True], [0.1, 0.9])
        self.assertEqual(evaluate([False, True], [0.1, 0.9], threshold)["f1"], 1.0)
        validate_queue_contract({name: {"positive": spec.positive, "negative": spec.negative} for name, spec in ATTACK_SPECS.items()})
        self.assertEqual(bottom_half_share([1.0, 2.0, 3.0, 4.0]), 0.3)
        transition = configured_transition()
        self.assertTrue(transition["activation_requires_commitment"])

    def test_gcs_governance_questionnaire(self):
        rows = pd.DataFrame({"context": ["game"] * 5, "policy_version": ["v1"] * 5, "schema_version": ["s1"] * 5, "split": ["fit"] * 5, "history": [5] * 5, "risk": [0.1] * 5, "mean": [0.5] * 5, "gini": [0.2] * 5, "dispersion": [0.1] * 5, "policy": [np.array([0.6, 0.19, 0.15, 0.06])] * 5, "signature": ["ok"] * 5})
        retained, policy = aggregate_context_summaries(rows, "game", signature_verifier=lambda row: row["signature"] == "ok", expected_policy_version="v1", expected_schema="s1")
        self.assertEqual(len(retained), 5)
        self.assertEqual(policy.shape, (4,))
        tally = tally_votes([{"id": "a", "weight": 10}, {"id": "b", "weight": 9}], [{"id": "a", "choice": "approve"}, {"id": "b", "choice": "reject"}], quorum=(1, 2), acceptance=(1, 2))
        self.assertTrue(tally["quorum_met"])
        lifecycle = GovernanceLifecycle()
        for state in (State.TRIGGERED, State.AUTHORIZED, State.CANDIDATE, State.VALIDATED, State.RATIFIED):
            lifecycle.transition(state)
        lifecycle.commit("v1")
        self.assertEqual(lifecycle.active_version, "v1")
        self.assertEqual(wilson_interval(37, 45)[0] < 37 / 45 < wilson_interval(37, 45)[1], True)

    def test_attack_injection_contract(self):
        negative_count = 1168
        metadata = pd.DataFrame({"streamer_id": [1] * negative_count, "cycle_id": [1] * negative_count, "entity_key": [f"u{i}" for i in range(negative_count)], "context": ["game"] * negative_count, "split": ["test"] * negative_count})
        raw = {name: np.ones((negative_count, 5), dtype=float) for name in RAW}
        features = np.full((negative_count, 5, 4), 0.25, dtype=float)
        clean = Windows(metadata, raw, features)
        fit = {"columns": {name: {"min": 0.0, "max": 100.0, "missing_median": 0.5} for name in ("watch_log", "comment", "like", "gift_amount", "activity_capped", "age_log", "device_price_score")}, "activity_p99": 100.0}
        fit["sybil_raw_template"] = {name: np.ones(5, dtype=float) for name in RAW}
        result = sybil(clean, fit, {"sybil_target_post_injection_fraction": 0.05}, np.random.default_rng(42), 42, 0)
        self.assertTrue(result.labels.any())

    def test_projection_is_feasible_and_fail_closed(self):
        projected, iterations = project_region(np.array([1.0, 0.0]), np.array([0.5, 0.5]), radius=0.2)
        self.assertGreater(iterations, 0)
        self.assertTrue(np.allclose(projected, [0.6, 0.4], atol=1e-10))
        with self.assertRaises(ValueError):
            project_region(np.array([np.nan, 0.0]), np.array([0.5, 0.5]), radius=0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
