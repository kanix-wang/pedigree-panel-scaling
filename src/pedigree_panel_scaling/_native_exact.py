"""Optional native execution of the same finite nuclear Bellman recurrence."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
import importlib
import importlib.util
from itertools import product
from numbers import Integral
from time import perf_counter
from types import MappingProxyType

from ._nuclear_exact import _FastModel, _MemoField, _PAIRS, _LOG_CPD, _solve_python


def available():
    """Inspect installed backend availability without loading the extension."""
    return importlib.util.find_spec(__package__+"._native") is not None


def _load_backend():
    try:
        return importlib.import_module(__package__+"._native")
    except ImportError:
        return None


def _catalog(model, tick):
    """Every supported signature reachable with at most the family's children."""
    children = len(model.children)
    records = []
    for profile in model.profiles:
        minimum = {}
        for count in range(children+1):
            for zero in range(count+1):
                for one in range(count-zero+1):
                    tick()
                    counts = zero, one, count-zero-one
                    likelihood = tuple(None if any(counts[x] and _LOG_CPD[i][x] is None for x in range(3))
                                       else sum(counts[x]*(_LOG_CPD[i][x] or 0) for x in range(3))
                                       for i in range(9))
                    for observed in product((-1, 0, 1, 2), repeat=2):
                        signature = tuple(exponent if all(observed[role] < 0 or observed[role] == _PAIRS[i][role]
                                                         for role in range(2)) else None
                                          for i, exponent in enumerate(likelihood))
                        if not any(exponent is not None and profile.prior[i] > 0
                                   for i, exponent in enumerate(signature)):
                            continue
                        identifier = profile.intern(signature)
                        minimum[identifier] = min(count, minimum.get(identifier, count))
        size = len(profile.nodes)
        if len(minimum) != size:
            raise RuntimeError("Finite locus catalog did not cover an initial posterior")
        profile_records = []
        for identifier in range(size):
            tick()
            node = profile.nodes[identifier]
            transitions = tuple(profile.successors(identifier, role)
                                if role < 2 or minimum[identifier] < children else ()
                                for role in range(3))
            profile_records.append((node.affine, node.risk, tuple(row[0] for row in node.genotype),
                                    node.constant, transitions))
        if len(profile.nodes) != size:
            raise RuntimeError("A finite-catalog transition created an unenumerated posterior")
        records.append(tuple(profile_records))
    return tuple(records)


@dataclass(frozen=True)
class _PackedMemo(Mapping):
    """Read-only lazy view; its capsule owns the completed native hash table."""

    backend: object
    capsule: object
    fields: tuple
    remaining_width: int
    max_remaining: int

    def __getitem__(self, key):
        if (not isinstance(key, tuple) or len(key) != len(self.fields)+2
                or any(isinstance(value, bool) or not isinstance(value, Integral) for value in key)):
            raise KeyError(key)
        mask, remaining, *identifiers = map(int, key)
        if not 0 <= mask <= 3 or not 0 <= remaining <= self.max_remaining:
            raise KeyError(key)
        packed = mask | (remaining << 2)
        for identifier, (shift, width) in zip(identifiers, self.fields):
            if identifier < 0 or identifier >= (1 << width):
                raise KeyError(key)
            packed |= identifier << shift
        result = self.backend.lookup(self.capsule, packed)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self):
        for packed in self.backend.keys(self.capsule):
            yield (packed & 3, (packed >> 2) & ((1 << self.remaining_width)-1),
                   *((packed >> shift) & ((1 << width)-1) for shift, width in self.fields))

    def __len__(self):
        return self.backend.size(self.capsule)


def solve_native(engine, observations=None, *, limits=None, progress=None, clock=perf_counter):
    """Run a complete exact solve; unavailable or wider backends use Python."""
    from .exact import ExactLimits, ExactLimitExceeded, ExactProgress, ExactSolution

    limits = ExactLimits() if limits is None else limits
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    started = clock()
    last_progress = started-1.0

    def emit(states=0, transitions=0, terminal=0, independent=0, elapsed=None):
        nonlocal last_progress
        now = clock()
        if progress is not None and now-last_progress >= 1.0:
            last_progress = now
            progress(ExactProgress(states, transitions, terminal, independent, now-started))

    def setup_tick():
        elapsed = clock()-started
        if limits.max_seconds is not None and elapsed >= limits.max_seconds:
            raise ExactLimitExceeded("max_seconds", 0, 0, elapsed)
        emit()

    def fallback():
        setup_tick()
        seconds = None if limits.max_seconds is None else limits.max_seconds-(clock()-started)
        if seconds is not None and seconds <= 0:
            raise ExactLimitExceeded("max_seconds", 0, 0, clock()-started)
        fallback_limits = ExactLimits(limits.max_states, limits.max_transitions, seconds)
        callback = None if progress is None else lambda snapshot: emit(
            snapshot.states_evaluated, snapshot.transitions_evaluated,
            snapshot.terminal_actions_closed, snapshot.independent_states_closed)
        try:
            solution = _solve_python(engine, observations, limits=fallback_limits, progress=callback, clock=clock)
        except ExactLimitExceeded as error:
            raise ExactLimitExceeded(error.reason, error.states_evaluated, error.transitions_evaluated,
                                     clock()-started) from error
        return replace(solution, elapsed_seconds=clock()-started)

    setup_tick()
    backend = _load_backend()
    setup_tick()
    if backend is None:
        return fallback()
    model = _FastModel(engine)
    root = engine._observations(engine.root_state if observations is None else observations)
    engine.state(root)
    root.setflags(write=False)
    root_key = model.key(root)
    profiles = _catalog(model, setup_tick)
    remaining_width = len(model.children).bit_length()
    shift = 2+remaining_width
    fields = []
    for profile, group in zip(model.profiles, model.groups):
        width = (len(profile.nodes)-1).bit_length()
        for _ in group:
            fields.append((shift, width))
            shift += width
    oversized_cap = any(cap is not None and cap > (1 << 64)-1
                        for cap in (limits.max_states, limits.max_transitions))
    if shift > 64 or oversized_cap:
        return fallback()
    setup_tick()
    seconds = None if limits.max_seconds is None else limits.max_seconds-(clock()-started)
    if seconds is not None and seconds <= 0:
        raise ExactLimitExceeded("max_seconds", 0, 0, clock()-started)
    payload = {"children": len(model.children), "fixed": model.fixed, "variable": model.variable,
               "profiles": profiles, "groups": tuple(len(group) for group in model.groups)}
    try:
        result = backend.solve(payload, root_key,
                               (limits.max_states, limits.max_transitions, seconds),
                               None if progress is None else emit)
    except backend.LimitExceeded as error:
        reason, states, transitions, _ = error.args
        raise ExactLimitExceeded(reason, states, transitions, clock()-started) from error
    scores = model.public_scores(result["root_role_values"], root)
    model.memo = _PackedMemo(backend, result["capsule"], tuple(fields), remaining_width, len(model.children))
    return ExactSolution(
        root_value=result["root_value"], root_action=max(scores, key=scores.__getitem__),
        root_action_values=MappingProxyType(scores), states_evaluated=result["states_evaluated"],
        transitions_evaluated=result["transitions_evaluated"],
        terminal_actions_closed=result["terminal_actions_closed"],
        independent_states_closed=result["independent_states_closed"], elapsed_seconds=clock()-started,
        _engine=engine, _root=root, _values=_MemoField(model.memo, 0),
        _actions=_MemoField(model.memo, 1), _codec=model,
        locus_states=sum(len(profile.nodes) for profile in model.profiles),
    )


__all__ = ["available", "solve_native"]
