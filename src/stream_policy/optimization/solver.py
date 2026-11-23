from __future__ import annotations

import json

import numpy as np

from ..allocation.cau import gini


LOSS_COEFFICIENTS = (1.0, 0.3, 0.1, 0.2, 0.2)


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.size == 0:
        return vector
    sorted_values = np.sort(vector)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, vector.size + 1)
    valid = sorted_values - cssv / indices > 0
    rho_index = indices[valid][-1] - 1 if valid.any() else 0
    theta = cssv[rho_index] / (rho_index + 1)
    return np.maximum(vector - theta, 0.0)


def _project_l1_ball(vector: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    delta = np.asarray(vector, dtype=float) - center
    if np.abs(delta).sum() <= radius:
        return np.asarray(vector, dtype=float).copy()
    magnitudes = np.sort(np.abs(delta))[::-1]
    cssv = np.cumsum(magnitudes)
    threshold = 0.0
    for index, magnitude in enumerate(magnitudes, start=1):
        candidate = (cssv[index - 1] - radius) / index
        if magnitude > candidate:
            threshold = candidate
    return center + np.sign(delta) * np.maximum(np.abs(delta) - threshold, 0.0)


def project_region(vector: np.ndarray, center: np.ndarray, radius: float = 0.8) -> tuple[np.ndarray, int]:
    x = np.asarray(vector, dtype=float).copy()
    center = np.asarray(center, dtype=float).copy()
    if x.ndim != 1 or center.ndim != 1 or x.shape != center.shape or x.size == 0:
        raise ValueError("projection vectors must be nonempty one-dimensional arrays of equal shape")
    if not np.isfinite(x).all() or not np.isfinite(center).all():
        raise ValueError("projection vectors must be finite")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("projection radius must be finite and nonnegative")
    if (center < -1e-12).any() or not np.isclose(center.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("authorized-region center must be a simplex policy")
    correction_simplex = np.zeros_like(x)
    correction_l1 = np.zeros_like(x)
    converged = False
    iterations = 0
    for iterations in range(1, 201):
        y = _project_simplex(x + correction_simplex)
        correction_simplex = x + correction_simplex - y
        z = _project_l1_ball(y + correction_l1, center, radius)
        correction_l1 = y + correction_l1 - z
        x = z
        if np.abs(x.sum() - 1.0) < 1e-12 and np.abs(x - _project_simplex(x)).sum() < 1e-10 and np.abs(x - center).sum() <= radius + 1e-10:
            converged = True
            break
    if not converged:
        raise ValueError("authorized-region projection did not converge")
    projected = _project_simplex(_project_l1_ball(x, center, radius))
    if (projected < -1e-10).any() or not np.isclose(projected.sum(), 1.0, rtol=0.0, atol=1e-10) or np.abs(projected - center).sum() > radius + 1e-8:
        raise ValueError("authorized-region projection returned an invalid policy")
    return projected, iterations


def allocate_policy(policy: np.ndarray, features: np.ndarray, budget: float = 1000.0) -> np.ndarray:
    score = np.maximum(np.asarray(features, dtype=float) @ np.asarray(policy, dtype=float), 0.0)
    total = score.sum()
    return score * budget / total if total > 0 else np.zeros(len(features), dtype=float)


def objective(policy: np.ndarray, current: np.ndarray, features: np.ndarray, risk: np.ndarray, budget: float = 1000.0) -> float:
    policy = np.asarray(policy, dtype=float)
    current = np.asarray(current, dtype=float)
    features = np.asarray(features, dtype=float)
    risk = np.asarray(risk, dtype=float)
    contribution = np.clip(features @ policy, 0.0, 1.0)
    allocation = allocate_policy(policy, features, budget)
    baseline_allocation = allocate_policy(current, features, budget)
    mean_verified = float(np.mean(contribution))
    risk_term = float(np.average(risk, weights=np.maximum(contribution, 1e-12)))
    gini_term = float(gini(allocation)) if allocation.sum() else 0.0
    allocation_displacement = float(np.abs(allocation - baseline_allocation).sum() / (2.0 * budget))
    policy_displacement = float(np.abs(policy - current).sum() / 2.0)
    return (1.0 - mean_verified) + 0.3 * risk_term + 0.1 * gini_term + 0.2 * allocation_displacement + 0.2 * policy_displacement


def finite_gradient(policy: np.ndarray, current: np.ndarray, features: np.ndarray, risk: np.ndarray, step: float = 0.0005, budget: float = 1000.0) -> np.ndarray:
    gradient = np.zeros_like(np.asarray(policy, dtype=float))
    for index in range(len(policy)):
        plus = np.asarray(policy, dtype=float).copy()
        minus = np.asarray(policy, dtype=float).copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus, current, features, risk, budget) - objective(minus, current, features, risk, budget)) / (2.0 * step)
    return gradient


def optimize_policy(
    initial: np.ndarray,
    current: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    risk_train: np.ndarray,
    risk_validation: np.ndarray,
    eta0: float = 0.08,
    eta_decay: float = 0.96,
    eta_min: float = 0.005,
    max_iterations: int = 100,
    allocation_tolerance: float = 0.01,
    policy_tolerance: float = 0.01,
    radius: float = 0.8,
    finite_difference_step: float = 0.0005,
    budget: float = 1000.0,
) -> tuple[list[dict], dict]:
    current = np.asarray(current, dtype=float)
    policy = np.asarray(initial, dtype=float).copy()
    train = np.asarray(train, dtype=float)
    validation = np.asarray(validation, dtype=float)
    risk_train = np.asarray(risk_train, dtype=float)
    risk_validation = np.asarray(risk_validation, dtype=float)
    if current.ndim != 1 or current.size == 0 or not np.isfinite(current).all() or (current < -1e-12).any() or not np.isclose(current.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("current policy must be a finite simplex vector")
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != current.size or validation.shape[1] != current.size or not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("training and validation features must be finite matrices matching the policy dimension")
    if risk_train.shape != (train.shape[0],) or risk_validation.shape != (validation.shape[0],) or not np.isfinite(risk_train).all() or not np.isfinite(risk_validation).all() or (risk_train < 0.0).any() or (risk_validation < 0.0).any():
        raise ValueError("risk vectors must match feature rows and be finite nonnegative")
    if type(max_iterations) is not int or max_iterations <= 0 or not np.isfinite(radius) or radius < 0.0:
        raise ValueError("invalid optimizer iteration or authorization settings")
    iteration_rows: list[dict] = []
    try:
        policy, initial_projection_iterations = project_region(policy, current, radius)
    except ValueError:
        policy = current.copy()
        initial_projection_iterations = None
    previous_allocation = allocate_policy(policy, train, budget)
    previous_policy = policy.copy()
    converged = False
    stop_reason = "max_iterations"
    candidate_rejected = initial_projection_iterations is None
    if candidate_rejected:
        stop_reason = "initial_projection_failed"
    else:
        for iteration in range(max_iterations):
            before = objective(policy, current, train, risk_train, budget)
            gradient = finite_gradient(policy, current, train, risk_train, finite_difference_step, budget)
            eta = max(eta_min, eta0 * (eta_decay ** iteration))
            proposal = policy - eta * gradient
            try:
                projected, projection_iterations = project_region(proposal, current, radius)
            except ValueError:
                stop_reason = "projection_failed"
                candidate_rejected = True
                break
            after = objective(projected, current, train, risk_train, budget)
            allocation = allocate_policy(projected, train, budget)
            allocation_delta = float(np.abs(allocation - previous_allocation).sum() / (2.0 * budget)) if budget else 0.0
            policy_delta = float(np.abs(projected - previous_policy).sum())
            validation_loss = objective(projected, current, validation, risk_validation, budget)
            projected_valid = bool(np.all(projected >= -1e-10) and np.isclose(projected.sum(), 1.0, atol=1e-10) and np.abs(projected - current).sum() <= radius + 1e-8)
            iteration_rows.append({
                "solver_iteration": iteration,
                "eta": eta,
                "loss_before": before,
                "loss_after": after,
                "allocation_delta": allocation_delta,
                "policy_delta": policy_delta,
                "validation_loss": validation_loss,
                "projection_iterations": projection_iterations,
                "simplex_violation": abs(float(projected.sum()) - 1.0),
                "authorization_violation": max(0.0, float(np.abs(projected - current).sum() - radius)),
                "projected_valid": projected_valid,
                "stop_reason": "",
            })
            if not projected_valid:
                stop_reason = "invalid_projection"
                candidate_rejected = True
                break
            policy = projected
            previous_allocation = allocation
            previous_policy = projected.copy()
            if allocation_delta < allocation_tolerance and policy_delta < policy_tolerance:
                converged = True
                stop_reason = "epsilon_A_and_epsilon_P"
                iteration_rows[-1]["stop_reason"] = stop_reason
                break
    validation_loss_final = objective(policy, current, validation, risk_validation, budget)
    validation_pass = bool(validation_loss_final <= objective(current, current, validation, risk_validation, budget) + 1e-12)
    if not converged or not validation_pass:
        candidate_rejected = True
        if stop_reason == "max_iterations" and not converged:
            stop_reason = "nonconvergent_candidate"
        elif not validation_pass and not stop_reason.endswith("failed"):
            stop_reason = "validation_failed"
        policy = current.copy()
    summary = {
        "initial_weights": json.dumps(np.asarray(initial, dtype=float).tolist()),
        "final_weights": json.dumps(policy.tolist()),
        "iterations": len(iteration_rows),
        "converged": converged,
        "stop_reason": stop_reason,
        "candidate_rejected": candidate_rejected,
        "train_loss_final": float(iteration_rows[-1]["loss_after"]) if iteration_rows else float(objective(current, current, train, risk_train, budget)),
        "validation_loss_final": float(validation_loss_final),
        "validation_pass": validation_pass,
        "final_budget_error": float(abs(allocate_policy(policy, validation, budget).sum() - budget)),
    }
    return iteration_rows, summary
