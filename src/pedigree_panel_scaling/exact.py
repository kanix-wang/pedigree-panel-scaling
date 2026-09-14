"""Full-horizon Bellman optimization with optional explicit resource limits.

This uses the same unrestricted joint-state recurrence as gt_code2's exact
backward induction. Every legal action and positive-probability panel is
included. Limits raise an exception; they never substitute a greedy policy.
Exactness is with respect to the model, using floating-point arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from itertools import product
import math
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

import numpy as np

from .inference import ObservationArray, PedigreeInference
from .exact_reductions import independent_scores


class _ExactCodec(Protocol):
    def key(self, observations: ObservationArray) -> Hashable: ...
    def decode_action(self, action: int, observations: ObservationArray) -> int: ...


class _ObservationCodec:
    """Preserve every person and genotype when specialized rules do not apply."""

    def __init__(self, engine: PedigreeInference) -> None:
        self.engine = engine

    def key(self, observations: ObservationArray) -> bytes:
        return self.engine._observations(observations).tobytes()

    def decode_action(self, action: int, observations: ObservationArray) -> int:
        if action == -1:
            return -1
        observed = self.engine._observations(observations)
        if not 0 <= action < self.engine.n_people or observed[0, action] >= 0:
            raise ValueError("Cached action must identify an untested person")
        return int(action)


@dataclass(frozen=True)
class ExactLimits:
    """Limits apply only when explicitly supplied; None means unlimited."""

    max_states: int | None = None
    max_transitions: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_states", "max_transitions"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.max_seconds is not None and (isinstance(self.max_seconds, bool)
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite")


@dataclass(frozen=True)
class ExactProgress:
    """Current work counts, without a partial value or optimality claim."""

    states_evaluated: int
    transitions_evaluated: int
    terminal_actions_closed: int
    independent_states_closed: int
    elapsed_seconds: float


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
    independent_states_closed: int
    elapsed_seconds: float
    _engine: PedigreeInference = field(repr=False, compare=False)
    _root: ObservationArray = field(repr=False, compare=False)
    _values: Mapping[Hashable, float] = field(repr=False, compare=False)
    _actions: Mapping[Hashable, int] = field(repr=False, compare=False)
    _codec: _ExactCodec = field(repr=False, compare=False)
    locus_states: int | None = None

    def _query(self, observations: ObservationArray | None) -> tuple[float, int]:
        observed = self._engine._observations(self._root if observations is None else observations)
        if np.any((self._root >= 0) & (observed != self._root)):
            raise ValueError("State must preserve the solved root's observations")
        key = self._codec.key(observed)
        if key in self._values:
            return self._values[key], self._codec.decode_action(self._actions[key], observed)
        # Independent continuations were closed analytically, including their
        # supported descendants. Queries validate support without new search.
        scores = independent_scores(self._engine, self._engine.state(observed))
        if scores is None:
            raise ValueError("State is outside the solved observation graph")
        action = max(scores, key=scores.__getitem__)
        return scores[action], action

    def value_at(self, observations: ObservationArray | None = None) -> float:
        return self._query(observations)[0]

    def action_at(self, observations: ObservationArray | None = None) -> int:
        return self._query(observations)[1]


def solve_exact(engine: PedigreeInference, observations: ObservationArray | None = None,
                *, limits: ExactLimits | None = None,
                progress: Callable[[ExactProgress], None] | None = None) -> ExactSolution:
    """Optimize the whole remaining testing strategy, or raise on a limit.

    Standard nuclear pedigrees recurse over exact parental-posterior states;
    other supported kernels retain the full public observation state.
    Time checks are cooperative between state/branch operations. State counts
    include entered decision states; terminal panels are not enumerated when
    only one person remains, since their continuation value is identically zero.
    No resource cap applies by default. An optional progress callback receives
    an initial snapshot, then at most one snapshot per second; no final callback
    is forced. The returned solution contains final counts. Callback exceptions
    propagate, and this function writes no output.
    """
    limits = ExactLimits() if limits is None else limits
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    # Delayed import keeps the backend dependent on the shared public result
    # and progress classes without an import cycle during package initialization.
    from ._nuclear_exact import solve_nuclear, supports_nuclear
    if supports_nuclear(engine):
        return solve_nuclear(engine, observations, limits=limits, progress=progress,
                             clock=perf_counter)
    started = perf_counter()
    last_progress = started - 1.0
    root = engine._observations(engine.root_state if observations is None else observations)
    root.setflags(write=False)
    codec = _ObservationCodec(engine)
    root_key = codec.key(root)
    values: dict[bytes, float] = {}
    actions: dict[bytes, int] = {}
    root_scores: dict[int, float] = {}
    states_count = transitions_count = terminal_closures = 0
    independent_closures = 0

    def check_time() -> None:
        nonlocal last_progress
        if limits.max_seconds is None and progress is None:
            return
        now = perf_counter()
        elapsed = now - started
        if limits.max_seconds is not None and elapsed >= limits.max_seconds:
            raise ExactLimitExceeded("max_seconds", states_count, transitions_count, elapsed)
        if progress is not None and now - last_progress >= 1.0:
            last_progress = now
            progress(ExactProgress(states_count, transitions_count, terminal_closures,
                                   independent_closures, elapsed))

    def visit(observed: ObservationArray) -> float:
        nonlocal states_count, transitions_count, terminal_closures, independent_closures
        check_time()
        key = codec.key(observed)
        if key in values:
            return values[key]
        if limits.max_states is not None and states_count >= limits.max_states:
            raise ExactLimitExceeded("max_states", states_count, transitions_count,
                                     perf_counter() - started)
        states_count += 1
        state = engine.state(observed)
        check_time()
        legal = np.flatnonzero(~state.tested)
        scores = independent_scores(engine, state)
        independent = scores is not None
        if independent:
            terminal_closures += int(len(legal) == 1)
            independent_closures += int(len(legal) > 1)
        else:
            scores = {-1: float(state.stop_reward)}
        for raw_person in (() if independent else legal):
            person = int(raw_person)
            check_time()
            immediate = float(state.test_rewards[person])
            support = [np.flatnonzero(state.genotype[g, person] > 0)
                       for g in range(engine.n_genes)]
            masses: list[float] = []
            terms: list[float] = []
            for panel in product(*support):
                check_time()
                if limits.max_transitions is not None and transitions_count >= limits.max_transitions:
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
        if key == root_key:
            root_scores.update(scores)
        check_time()
        return values[key]

    value = visit(root)
    return ExactSolution(
        root_value=value, root_action=codec.decode_action(actions[root_key], root),
        root_action_values=MappingProxyType(root_scores),
        states_evaluated=states_count, transitions_evaluated=transitions_count,
        terminal_actions_closed=terminal_closures,
        independent_states_closed=independent_closures,
        elapsed_seconds=perf_counter() - started, _engine=engine, _root=root,
        _values=MappingProxyType(values), _actions=MappingProxyType(actions), _codec=codec,
    )


__all__ = ["ExactLimits", "ExactLimitExceeded", "ExactProgress", "ExactSolution", "solve_exact"]
