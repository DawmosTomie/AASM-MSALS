import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_rq2
import run_rq3
import run_rq4
import run_rq6


class RunnerTests(unittest.TestCase):
    def test_rq2_executes_continuous_baselines_and_cau(self):
        frame = pd.DataFrame([
            {"streamer_id": "s", "user_id": "a", "cycle_id": 1, "context": "game", "new_budget": 100.0, "click": 0.0, "watch_ms": 0.0, "engagement": 0.0, "financial": 0.0, "activity": 0.0, "authenticity": 0.0},
            {"streamer_id": "s", "user_id": "b", "cycle_id": 1, "context": "game", "new_budget": 100.0, "click": 0.0, "watch_ms": 0.0, "engagement": 0.0, "financial": 0.0, "activity": 0.0, "authenticity": 0.0},
            {"streamer_id": "s", "user_id": "a", "cycle_id": 2, "context": "game", "new_budget": 50.0, "click": 2.0, "watch_ms": 1000.0, "engagement": 1.0, "financial": 0.0, "activity": 0.0, "authenticity": 0.0},
            {"streamer_id": "s", "user_id": "b", "cycle_id": 2, "context": "game", "new_budget": 50.0, "click": 1.0, "watch_ms": 1000.0, "engagement": 0.0, "financial": 1.0, "activity": 0.0, "authenticity": 0.0},
        ])
        fit = {
            "traffic": {"activity_p99": 10.0, "click": {"min": 0.0, "max": 10.0}, "watch_log": {"min": 0.0, "max": float(np.log1p(1000.0))}},
            "reference_4d": np.zeros((4, 4)).tolist(),
        }
        config_path = ROOT / "config" / "protocol.toml"
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            input_path = temp_path / "allocation.csv"
            fit_path = temp_path / "fit.json"
            frame.to_csv(input_path, index=False)
            fit_path.write_text(json.dumps(fit), encoding="utf-8")
            result = run_rq2.run(input_path, fit_path, config_path)
        self.assertIn("raw_traffic", result["methods"])
        self.assertIn("cau", result["methods"])
        self.assertEqual(result["methods"]["cau"]["zero_total_cells"], 1)

    def test_rq3_requires_and_executes_three_initialization_arms(self):
        payload = {
            "current": [0.6, 0.19, 0.15, 0.06],
            "same_context_mean": [0.55, 0.25, 0.14, 0.06],
            "gcs_weighted": [0.50, 0.30, 0.14, 0.06],
            "train": [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0]],
            "validation": [[0.7, 0.2, 0.1, 0.0], [0.2, 0.7, 0.1, 0.0]],
            "risk_train": [0.1, 0.2],
            "risk_validation": [0.1, 0.2],
        }
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            input_path = temp_path / "optimizer.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            result = run_rq3.run(input_path, ROOT / "config" / "protocol.toml")
        self.assertEqual(set(result), {"previous_local", "same_context_mean", "gcs_weighted"})
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            incomplete = dict(payload)
            del incomplete["gcs_weighted"]
            input_path = temp_path / "incomplete.json"
            input_path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaises(KeyError):
                run_rq3.run(input_path, ROOT / "config" / "protocol.toml")

    def test_rq4_executes_three_ablation_metrics(self):
        payload = {
            "budget_minor_units": 100,
            "cells": [{
                "clean_scores": {"a": 2, "b": 1},
                "full_scores": {"a": 2, "b": 1},
                "no_reliability_scores": {"a": 1, "b": 2},
                "full_recovery_cycles": [2, 2],
                "no_cross_tenant_recovery_cycles": [3, 4],
                "full_committed_policies": [[0.5, 0.5], [0.55, 0.45]],
                "no_authorization_committed_policies": [[0.5, 0.5], [0.8, 0.2]],
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ablation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = run_rq4.run(path)
        metrics = result["ablation_metrics"]
        self.assertEqual(set(metrics), {"reliability_adjustment", "cross_tenant_initialization", "authorization_constraints"})
        self.assertAlmostEqual(metrics["cross_tenant_initialization"]["full_mean_recovery_cycles"], 2.0)
        self.assertGreater(metrics["authorization_constraints"]["no_authorization_mean_committed_weight_variation"], metrics["authorization_constraints"]["full_mean_committed_weight_variation"])

    def test_rq6_reports_b1_to_b4_wilson_intervals(self):
        rows = []
        for index in range(45):
            rows.append({
                "participant_id": index,
                "role_group": ("Streamer", "Viewer", "Validator")[index % 3],
                "presentation_order": "A-B" if index < 24 else "B-A",
                **{f"Q{item}": 4 for item in range(1, 6)},
                **{f"B{item}": ("A", "B", "Same/No Difference")[index % 3] for item in range(1, 5)},
            })
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "responses.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = run_rq6.run(path)
        self.assertEqual(set(result["preferences"]), {"B1", "B2", "B3", "B4"})
        self.assertIn("wilson", result["preferences"]["B1"])
        self.assertEqual(result["likert"]["Q1"]["valid_n"], 45)


if __name__ == "__main__":
    unittest.main()
