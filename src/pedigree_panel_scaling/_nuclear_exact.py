"""Full Bellman recursion on exact nuclear posterior identifiers.

No observation arrays or pedigree recalibrations occur in recursive branches.
All supported complete-panel outcomes remain in the Bellman expectation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
import math
from time import perf_counter
from types import MappingProxyType

import numpy as np


_PAIRS = tuple(product(range(3), repeat=2))
_CPD = tuple(((1-a/2)*(1-b/2), a/2*(1-b/2)+(1-a/2)*b/2, a*b/4)
             for a, b in _PAIRS)
_LOG_CPD = tuple(tuple(None if p == 0 else -2 if p == .25 else -1 if p == .5 else 0
                       for p in row) for row in _CPD)


@dataclass(slots=True)
class _Locus:
    signature: tuple
    genotype: tuple
    affine: tuple
    risk: tuple
    constant: tuple
    transitions: list = field(default_factory=lambda: [None, None, None])


class _Profile:
    """Intern exact relative likelihood signatures for a single gene profile."""

    def __init__(self, frequency, coefficients):
        founder = ((1-frequency)**2, 2*frequency*(1-frequency), frequency**2)
        self.prior = tuple(founder[a]*founder[b] for a, b in _PAIRS)
        self.coefficients = coefficients
        self.ids = {}
        self.nodes = []

    def intern(self, signature):
        signature = tuple(exponent if self.prior[i] > 0 else None
                          for i, exponent in enumerate(signature))
        finite = [exponent for exponent in signature if exponent is not None]
        if not finite:
            raise ValueError("Evidence has zero support under the pedigree model")
        largest = max(finite)
        signature = tuple(None if exponent is None else exponent-largest for exponent in signature)
        existing = self.ids.get(signature)
        if existing is not None:
            return existing
        weight = tuple(0.0 if exponent is None else self.prior[i]*math.ldexp(1.0, exponent)
                       for i, exponent in enumerate(signature))
        if any(exponent is not None and weight[i] <= 0 for i, exponent in enumerate(signature)):
            raise FloatingPointError("Positive nuclear posterior weight underflowed")
        total = math.fsum(weight)
        posterior = tuple(value/total for value in weight)
        if any(weight[i] > 0 and posterior[i] <= 0 for i in range(9)):
            raise FloatingPointError("Positive nuclear posterior probability underflowed")
        genotype = []
        for role in range(3):
            if role < 2:
                probabilities = tuple(math.fsum(posterior[i] for i in range(9) if _PAIRS[i][role] == x)
                                      for x in range(3))
            else:
                probabilities = tuple(math.fsum(posterior[i]*_CPD[i][x] for i in range(9))
                                      for x in range(3))
            genotype.append(probabilities)
        carrier = tuple(min(1.0, max(0.0, math.fsum(row[1:]))) for row in genotype)
        affine = tuple(a*(1-delta)*p+omega
                       for (a, b, delta, omega), p in zip(self.coefficients, carrier))
        risk = tuple(-(a*delta+b)*p*(1-p)
                     for (a, b, delta, omega), p in zip(self.coefficients, carrier))
        locus = _Locus(signature, tuple(genotype), affine, risk,
                       tuple(sum(p > 0 for p in row) == 1 for row in genotype))
        identifier = len(self.nodes)
        self.ids[signature] = identifier
        self.nodes.append(locus)
        return identifier

    def successors(self, identifier, role):
        locus = self.nodes[identifier]
        cached = locus.transitions[role]
        if cached is not None:
            return cached
        successors = []
        for outcome, probability in enumerate(locus.genotype[role]):
            if probability <= 0:
                continue
            if role < 2:
                signature = tuple(exponent if _PAIRS[i][role] == outcome else None
                                  for i, exponent in enumerate(locus.signature))
            else:
                signature = tuple(None if exponent is None or _LOG_CPD[i][outcome] is None
                                  else exponent+_LOG_CPD[i][outcome]
                                  for i, exponent in enumerate(locus.signature))
            successors.append((probability, self.intern(signature)))
        if abs(math.fsum(p for p, _ in successors)-1.0) > 1e-12:
            raise FloatingPointError("Nuclear outcome probabilities do not sum to one")
        result = tuple(successors)
        locus.transitions[role] = result
        return result


def supports_nuclear(engine):
    """Dispatch only on the captured standard Mendelian kernel and rewards."""
    from .nuclear import NuclearInference

    if type(engine) is not NuclearInference:
        return False
    if not (np.array_equal(engine._parent_genotypes, _PAIRS)
            and np.array_equal(engine._child_cpd, _CPD)):
        return False
    children = engine.child_indices
    if any(not np.all(parameter[:, children] == parameter[:, children[0], None])
           for parameter in (engine.a, engine.b, engine.delta, engine.omega)):
        return False
    return np.array_equal(engine.risk_weights, -(engine.a*engine.delta+engine.b))


@dataclass(frozen=True)
class _MemoField(Mapping):
    """Read-only field view over shared (value, action-role) memo entries."""

    records: Mapping
    position: int

    def __getitem__(self, key):
        return self.records[key][self.position]

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


class _FastModel:
    def __init__(self, engine):
        if not supports_nuclear(engine):
            raise ValueError("Nuclear optimization requires the standard Mendelian kernel and exchangeable children")
        self.engine = engine
        self.parents = tuple(engine.parent_indices)
        self.children = tuple(engine.child_indices)
        profile_groups = {}
        for gene in range(engine.n_genes):
            profile = (float(engine.frequencies[gene]), *(float(value)
                       for parameter in (engine.a, engine.b, engine.delta, engine.omega)
                       for value in parameter[gene]))
            profile_groups.setdefault(profile, []).append(gene)
        self.groups = tuple(tuple(indices) for indices in profile_groups.values())
        self.profiles = []
        self.by_gene = [None]*engine.n_genes
        self.flat_profiles = []
        self.spans = []
        offset = 0
        for group in self.groups:
            gene = group[0]
            coefficients = tuple(tuple(float(parameter[gene, person]) for parameter in
                                       (engine.a, engine.b, engine.delta, engine.omega))
                                 for person in (*self.parents, self.children[0]))
            profile = _Profile(float(engine.frequencies[gene]), coefficients)
            self.profiles.append(profile)
            for gene in group:
                self.by_gene[gene] = profile
            self.flat_profiles.extend([profile]*len(group))
            self.spans.append((offset, offset+len(group)))
            offset += len(group)
        self.fixed = float(engine.case.fixed_cost)
        self.variable = float(engine.case.variable_cost)
        self.memo = {}

    def canonical(self, mask, remaining, identifiers):
        result = [mask, remaining]
        for start, stop in self.spans:
            if stop-start == 1:
                result.append(identifiers[start])
            else:
                result.extend(sorted(identifiers[start:stop]))
        return tuple(result)

    def key(self, observations):
        observed = self.engine._observations(observations)
        identifiers = []
        for group in self.groups:
            for gene in group:
                counts = tuple(int(np.count_nonzero(observed[gene, self.children] == x)) for x in range(3))
                signature = []
                for i, pair in enumerate(_PAIRS):
                    if any(observed[gene, self.parents[role]] >= 0
                           and observed[gene, self.parents[role]] != pair[role] for role in range(2)):
                        signature.append(None)
                    elif any(counts[x] and _LOG_CPD[i][x] is None for x in range(3)):
                        signature.append(None)
                    else:
                        signature.append(sum(counts[x]*(_LOG_CPD[i][x] or 0) for x in range(3)))
                identifiers.append(self.by_gene[gene].intern(signature))
        mask = sum((1 << role) for role in range(2) if observed[0, self.parents[role]] >= 0)
        remaining = int(np.count_nonzero(observed[0, self.children] < 0))
        return self.canonical(mask, remaining, identifiers)

    def summary(self, key):
        mask, remaining = key[:2]
        nodes = tuple(profile.nodes[identifier] for profile, identifier in zip(self.flat_profiles, key[2:]))
        roles = [role for role in range(2) if not mask & (1 << role)]
        if remaining:
            roles.append(2)
        affine = tuple(math.fsum(node.affine[role] for node in nodes) for role in range(3))
        risk = tuple(math.fsum(node.risk[role] for node in nodes) for role in range(3))
        stop = tuple(a-r for a, r in zip(affine, risk))
        test = tuple(affine[role]-self.fixed-self.variable*(1-math.prod(node.genotype[role][0] for node in nodes))
                     for role in range(3))
        counts = (int(not mask & 1), int(not mask & 2), remaining)
        scores = {-1: math.fsum(counts[role]*stop[role] for role in roles)}
        independent = all((node.constant[0] or node.constant[1])
                          and (not remaining or ((node.constant[0] or node.constant[2])
                                                 and (node.constant[1] or node.constant[2])))
                          for node in nodes)
        if independent:
            best = tuple(max(stop[role], test[role]) for role in range(3))
            for chosen in roles:
                scores[chosen] = math.fsum([test[chosen], *(
                    (counts[role]-int(role == chosen))*best[role] for role in roles)])
        return roles, test, scores, independent

    def decode_action(self, role, observations):
        if role < 0:
            return -1
        if role < 2:
            person = self.parents[role]
            if observations[0, person] >= 0:
                raise ValueError("Cached parent action is already tested")
            return person
        for child in self.children:
            if observations[0, child] < 0:
                return child
        raise ValueError("Cached child action has no untested child")

    def public_scores(self, scores, observations):
        result = {-1: scores[-1]}
        for person in range(self.engine.n_people):
            if observations[0, person] < 0:
                role = self.parents.index(person) if person in self.parents else 2
                result[person] = scores[role]
        return result


def _solve_python(engine, observations=None, *, limits=None, progress=None, clock=perf_counter):
    """Return a complete nuclear-family optimum, with no default resource cap."""
    from .exact import ExactLimits, ExactLimitExceeded, ExactProgress, ExactSolution

    limits = ExactLimits() if limits is None else limits
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    started = clock()
    last_progress = started-1.0
    root = engine._observations(engine.root_state if observations is None else observations)
    engine.state(root)
    root.setflags(write=False)
    model = _FastModel(engine)
    root_key = model.key(root)
    states = transitions = terminal = independent_count = 0
    root_scores = {}

    def tick():
        nonlocal last_progress
        if limits.max_seconds is None and progress is None:
            return
        now = clock()
        elapsed = now-started
        if limits.max_seconds is not None and elapsed >= limits.max_seconds:
            raise ExactLimitExceeded("max_seconds", states, transitions, elapsed)
        if progress is not None and now-last_progress >= 1.0:
            last_progress = now
            progress(ExactProgress(states, transitions, terminal, independent_count, elapsed))

    def visit(key):
        nonlocal states, transitions, terminal, independent_count
        tick()
        cached = model.memo.get(key)
        if cached is not None:
            return cached[0]
        if limits.max_states is not None and states >= limits.max_states:
            raise ExactLimitExceeded("max_states", states, transitions, clock()-started)
        states += 1
        roles, immediate, scores, independent = model.summary(key)
        mask, remaining = key[:2]
        if independent:
            legal_count = 2-mask.bit_count()+remaining
            terminal += int(legal_count == 1)
            independent_count += int(legal_count > 1)
        else:
            for role in roles:
                alternatives = [profile.successors(identifier, role)
                                for profile, identifier in zip(model.flat_profiles, key[2:])]
                next_mask = mask | (1 << role) if role < 2 else mask
                next_remaining = remaining-int(role == 2)
                masses, terms = [], []
                for outcomes in product(*alternatives):
                    tick()
                    if limits.max_transitions is not None and transitions >= limits.max_transitions:
                        raise ExactLimitExceeded("max_transitions", states, transitions, clock()-started)
                    probability = math.prod(outcome[0] for outcome in outcomes)
                    if probability <= 0 or not math.isfinite(probability):
                        raise FloatingPointError("Positive complete-panel probability underflowed or is nonfinite")
                    transitions += 1
                    successor = model.canonical(next_mask, next_remaining, tuple(outcome[1] for outcome in outcomes))
                    terms.append(probability*visit(successor))
                    masses.append(probability)
                if abs(math.fsum(masses)-1.0) > 1e-10:
                    raise FloatingPointError("Complete-panel probabilities do not sum to one")
                scores[role] = immediate[role]+math.fsum(terms)
        if not all(math.isfinite(value) for value in scores.values()):
            raise FloatingPointError("Nonfinite Bellman action value")
        role = max(scores, key=scores.__getitem__)
        model.memo[key] = scores[role], role
        if key == root_key:
            root_scores.update(model.public_scores(scores, root))
        tick()
        return scores[role]

    root_value = visit(root_key)
    root_action = max(root_scores, key=root_scores.__getitem__)
    model.memo = MappingProxyType(model.memo)
    return ExactSolution(
        root_value=root_value, root_action=root_action,
        root_action_values=MappingProxyType(root_scores), states_evaluated=states,
        transitions_evaluated=transitions, terminal_actions_closed=terminal,
        independent_states_closed=independent_count, elapsed_seconds=clock()-started,
        _engine=engine, _root=root, _values=_MemoField(model.memo, 0),
        _actions=_MemoField(model.memo, 1), _codec=model,
        locus_states=sum(len(profile.nodes) for profile in model.profiles),
    )


def solve_nuclear(engine, observations=None, *, limits=None, progress=None, clock=perf_counter):
    """Use optional native recursion when installed, otherwise exact Python."""
    from ._native_exact import available, solve_native

    solver = solve_native if available() else _solve_python
    return solver(engine, observations, limits=limits, progress=progress, clock=clock)


__all__ = ["solve_nuclear", "supports_nuclear"]
