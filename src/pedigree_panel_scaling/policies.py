"""Two existing policies using only currently observed complete panels."""

from __future__ import annotations

import numpy as np

from .inference import ObservationArray, PedigreeInference
from .nuclear import NuclearInference


def choose_action(engine: PedigreeInference, observations: ObservationArray,
                  policy: str = "information_greedy") -> int:
    """Return the next person's index, or -1 for STOP.

    Recursive myopic compares immediate TEST rewards to STOP. Information
    greedy uses exact expected risk reduction from one test, minus panel cost.
    Both preserve the respective source policy's tie rule and people order.
    """
    if policy not in ("recursive_myopic", "information_greedy"):
        raise ValueError(f"Unknown policy: {policy!r}")
    with_pairs = policy == "information_greedy" and not isinstance(engine, NuclearInference)
    state = engine.state(observations, with_pairs=with_pairs)
    legal = np.flatnonzero(~state.tested)
    if not len(legal):
        return -1
    if policy == "recursive_myopic":
        chosen, best = -1, float(state.stop_reward)
        for person in legal:
            value = float(state.test_rewards[person])
            if value > best + 1e-6:
                chosen, best = int(person), value
        return chosen
    if isinstance(engine, NuclearInference):
        gains = engine.rh1_gains_batch(state.observations[None], state.test_costs[None])[0]
    else:
        gains = state.myopic_gains
    if gains is None:
        raise RuntimeError("Information scores are missing")
    actions = [-1, *(int(person) for person in legal)]
    scores = [float(state.stop_reward),
              *(float(state.stop_reward + gains[person]) for person in legal)]
    best = max(scores)
    for action, value in zip(actions, scores):
        if best - value <= 1e-10 + 1e-10 * max(1.0, abs(best), abs(value)):
            return action
    raise RuntimeError("No finite action score")
