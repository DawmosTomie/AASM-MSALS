from __future__ import annotations

import numpy as np

from ..optimization.solver import project_region


GAMING_POLICY = np.array([0.60, 0.19, 0.15, 0.06], dtype=float)
SHOPPING_POLICY = np.array([0.24, 0.55, 0.15, 0.06], dtype=float)


def configured_transition(trigger_event: int = 32, radius: float = 0.40) -> dict[str, object]:
    if type(trigger_event) is not int or trigger_event < 0:
        raise ValueError("trigger_event must be a nonnegative integer")
    if type(radius) not in (int, float) or radius < 0:
        raise ValueError("radius must be nonnegative")
    candidate, iterations = project_region(SHOPPING_POLICY, GAMING_POLICY, float(radius))
    return {"from": GAMING_POLICY.copy(), "target": SHOPPING_POLICY.copy(), "candidate": candidate, "trigger_event": trigger_event, "radius": float(radius), "projection_iterations": iterations, "activation_requires_commitment": True}
