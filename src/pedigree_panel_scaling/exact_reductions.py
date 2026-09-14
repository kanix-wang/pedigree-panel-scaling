"""Exact closure of conditionally independent testing decisions."""

from __future__ import annotations

import math

import numpy as np

from .inference import InferenceState, PedigreeInference


def independent_scores(engine: PedigreeInference, state: InferenceState) -> dict[int, float] | None:
    """Full Bellman action values when untested people are independent.

    ``state`` must be the engine's supported state for these observations.
    Each moral edge must have a constant endpoint in every gene. Support size,
    rather than a rounded maximum probability, determines whether it is a
    constant. The shared within-panel cost remains intact.
    """
    constant = np.count_nonzero(state.genotype > 0, axis=-1) == 1
    if any(not np.all(constant[:, a] | constant[:, b]) for a, b in engine.moral_edges):
        return None
    legal = [int(person) for person in np.flatnonzero(~state.tested)]
    affine = engine.a * (1 - engine.delta) * state.carrier + engine.omega
    risk = engine.risk_weights * state.carrier * (1 - state.carrier)
    stop = {
        person: math.fsum(float(value) for value in affine[:, person])
        - math.fsum(float(value) for value in risk[:, person])
        for person in legal
    }
    test = {person: float(state.test_rewards[person]) for person in legal}
    best = {person: max(stop[person], test[person]) for person in legal}
    scores = {-1: math.fsum(stop.values())}
    for person in legal:
        scores[person] = math.fsum([test[person], *(best[other] for other in legal if other != person)])
    return scores


__all__ = ["independent_scores"]
