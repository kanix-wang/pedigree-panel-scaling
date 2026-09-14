"""Full-horizon Bellman optimization with explicit resource limits.

This uses the same unrestricted joint-state recurrence as gt_code2's exact
backward induction. Every legal action and positive-probability panel is
included. Limits raise an exception; they never substitute a greedy policy.
Exactness is with respect to the model, using floating-point arithmetic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
import math
from time import perf_counter
from types import MappingProxyType

import numpy as np

from .inference import ObservationArray, PedigreeInference


@dataclass(frozen=True)
class ExactLimits:
    max_states: int = 100_000
    max_transitions: int = 2_000_000
    max_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("max_states", "max_transitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.max_seconds, bool)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")


class ExactLimitExceeded(RuntimeError):
    """Optimization stopped without producing an exact solution."""

    def __init__(self, reason: str, states_evaluated: int,
                 transitions_evaluated: int, elapsed_seconds: float) -> None:
        self.reason = reason
        self.states_evaluated = states_evaluated
        self.transitions_evaluated = transitions_evaluated
        self.elapsed_seconds = elapsed_seconds
        super().__init__(f"Exact optimization exceeded {reason}; no exact solution returned")


@dataclass(frozen=True)
class ExactSolution:
    root_value: float
    root_action: int
    root_action_values: Mapping[int, float]
    states_evaluated: int
    transitions_evaluated: int
    terminal_actions_closed: int
    elapsed_seconds: float
    _engine: PedigreeInference = field(repr=False, compare=False)
    _root: ObservationArray = field(repr=False, compare=False)
    _values: Mapping[bytes, float] = field(repr=False, compare=False)
    _actions: Mapping[bytes, int] = field(repr=False, compare=False)

    def _query(self, observations: ObservationArray | None) -> tuple[bytes, bool]:
        observed = self._engine._observations(self._root if observations is None else observations)
        if np.any((self._root >= 0) & (observed != self._root)):
            raise ValueError("State must preserve the solved root's observations")
        key = observed.tobytes()
        terminal = bool(np.all(observed >= 0))
        if key not in self._values:
            if not terminal:
                raise ValueError("State is outside the solved observation graph")
            # Fully tested successors were closed analytically; validate support.
            self._engine.state(observed)
        return key, terminal

    def value_at(self, observations: ObservationArray | None = None) -> float:
        key, terminal = self._query(observations)
        return 0.0 if terminal else self._values[key]

    def action_at(self, observations: ObservationArray | None = None) -> int:
        key, terminal = self._query(observations)
        return -1 if terminal else self._actions[key]


def solve_exact(engine: PedigreeInference, observations: ObservationArray | None = None,
                *, limits: ExactLimits | None = None) -> ExactSolution:
    """Optimize the whole remaining testing strategy, or raise on a limit.

    Time checks are cooperative between state/branch operations. State counts
    include entered decision states; terminal panels are not enumerated when
    only one person remains, since their continuation value is identically zero.
    """
    limits = ExactLimits() if limits is None else limits
    started = perf_counter()
    root = engine._observations(engine.root_state if observations is None else observations)
    root.setflags(write=False)
    values: dict[bytes, float] = {}
    actions: dict[bytes, int] = {}
    root_scores: dict[int, float] = {}
    states_count = transitions_count = terminal_closures = 0

    def check_time() -> None:
        elapsed = perf_counter() - started
        if elapsed >= limits.max_seconds:
            raise ExactLimitExceeded("max_seconds", states_count, transitions_count, elapsed)

    def visit(observed: ObservationArray) -> float:
        nonlocal states_count, transitions_count, terminal_closures
        check_time()
        key = observed.tobytes()
        if key in values:
            return values[key]
        if states_count >= limits.max_states:
            raise ExactLimitExceeded("max_states", states_count, transitions_count,
                                     perf_counter() - started)
        states_count += 1
        state = engine.state(observed)
        check_time()
        legal = np.flatnonzero(~state.tested)
        scores = {-1: float(state.stop_reward)}
        for raw_person in legal:
            person = int(raw_person)
            check_time()
            immediate = float(state.test_rewards[person])
            if len(legal) == 1:
                # No unresolved people remain after this test; V(successor)=0.
                scores[person] = immediate
                terminal_closures += 1
                continue
            support = [np.flatnonzero(state.genotype[g, person] > 0)
                       for g in range(engine.n_genes)]
            masses: list[float] = []
            terms: list[float] = []
            for panel in product(*support):
                check_time()
                if transitions_count >= limits.max_transitions:
                    raise ExactLimitExceeded("max_transitions", states_count, transitions_count,
                                             perf_counter() - started)
                probability = math.prod(float(state.genotype[g, person, value])
                                        for g, value in enumerate(panel))
                if probability <= 0 or not math.isfinite(probability):
                    raise FloatingPointError("Positive panel probability underflowed or is nonfinite")
                transitions_count += 1
                successor = engine.observe(observed, person, np.asarray(panel, dtype=np.int8))
                terms.append(probability * visit(successor))
                masses.append(probability)
            if abs(math.fsum(masses) - 1.0) > 1e-10:
                raise FloatingPointError("Enumerated panel probabilities do not sum to one")
            scores[person] = immediate + math.fsum(terms)
        if not all(math.isfinite(value) for value in scores.values()):
            raise FloatingPointError("Nonfinite Bellman action value")
        # Strict maximization, with STOP then people order for exact ties.
        action = max(scores, key=scores.__getitem__)
        values[key], actions[key] = scores[action], action
        if key == root.tobytes():
            root_scores.update(scores)
        check_time()
        return values[key]

    value = visit(root)
    return ExactSolution(
        root_value=value, root_action=actions[root.tobytes()],
        root_action_values=MappingProxyType(root_scores),
        states_evaluated=states_count, transitions_evaluated=transitions_count,
        terminal_actions_closed=terminal_closures,
        elapsed_seconds=perf_counter() - started, _engine=engine, _root=root,
        _values=MappingProxyType(values), _actions=MappingProxyType(actions),
    )


__all__ = ["ExactLimits", "ExactLimitExceeded", "ExactSolution", "solve_exact"]
