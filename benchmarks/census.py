"""Integer nuclear-state/branch census; never runs an optimizer.

Two parents, exchangeable children, no evidence, positive numerical support,
independent perfect panels, and the current exact support-closure rule only.
Adapted from the independently checked 2026-09-14 capacity-study census.
"""
from functools import lru_cache
from itertools import product
from math import comb, prod

PAIRS = tuple(product(range(3), repeat=2))
CPD4 = tuple(((2-a)*(2-b), a*(2-b)+(2-a)*b, a*b) for a, b in PAIRS)
LOG_CPD = tuple(tuple({0: None, 1: -2, 2: -1, 4: 0}[p] for p in row) for row in CPD4)


@lru_cache(None)
def signatures(k, mask):
    found = set()
    roles = [r for r in range(2) if mask & (1 << r)]
    for zero in range(k+1):
        for one in range(k-zero+1):
            counts = zero, one, k-zero-one
            for values in product(range(3), repeat=len(roles)):
                observed = dict(zip(roles, values))
                row = [None if any(pair[r] != x for r, x in observed.items())
                       or any(c and e is None for c, e in zip(counts, logs))
                       else sum(c*(e or 0) for c, e in zip(counts, logs))
                       for pair, logs in zip(PAIRS, LOG_CPD)]
                finite = [e for e in row if e is not None]
                if finite:
                    largest = max(finite)
                    found.add(tuple(None if e is None else e-largest for e in row))
    return frozenset(found)


def groups_for(genes, profile):
    if profile not in ('distinct', 'alternating') or genes < 1:
        raise ValueError('Positive gene count and distinct/alternating profiles required')
    return (1,)*genes if profile == 'distinct' else tuple(r for r in ((genes+1)//2, genes//2) if r)


def state_count(people, genes, profile='distinct'):
    if people < 3:
        raise ValueError('Nuclear census requires two parents and at least one child')
    groups = groups_for(genes, profile)
    total = 0
    for k in range(people-1):
        for a, multiplicity in enumerate((1, 2, 1)):
            if a == 2 and k == people-2:
                continue
            classes = (9 if a == 2 else (1 if k == 0 else 3 if k == 1 else 2*k+2)
                       if a == 0 else (3 if k == 0 else 7 if k == 1 else 2*k+8))
            total += multiplicity * prod(comb(classes+r-1, r) for r in groups)
    return total


@lru_cache(None)
def spectra(k, mask, children_remain, degree):
    all_weights, closed_weights = [[], [], []], [[], [], []]
    closed_count = 0
    for signature in signatures(k, mask):
        support = [i for i, e in enumerate(signature) if e is not None]
        sizes = [len({PAIRS[i][r] for i in support}) for r in range(2)]
        sizes.append(len({x for i in support for x in range(3) if CPD4[i][x]}))
        f, m, c = (s == 1 for s in sizes)
        closed = (f or m) and (not children_remain or ((f or c) and (m or c)))
        closed_count += closed
        for role, weight in enumerate(sizes):
            all_weights[role].append(weight)
            if closed:
                closed_weights[role].append(weight)

    def homogeneous(weights):
        h = [1] + [0]*degree
        for weight in weights:
            for r in range(1, degree+1):
                h[r] += weight*h[r-1]
        return h
    return ([homogeneous(w) for w in all_weights],
            [homogeneous(w) for w in closed_weights], closed_count)


def census(people, genes, profile='distinct'):
    states = state_count(people, genes, profile)
    groups = groups_for(genes, profile)
    branches = terminal = independent = 0
    for k in range(people-1):
        remaining = people-2-k
        for mask in range(4):
            all_h, closed_h, closed = spectra(k, mask, bool(remaining), max(groups))
            roles = [r for r in range(2) if not mask & (1 << r)] + ([2] if remaining else [])
            for role in roles:
                branches += prod(all_h[role][r] for r in groups) - prod(closed_h[role][r] for r in groups)
            legal = remaining+2-mask.bit_count()
            count = prod(comb(closed+r-1, r) for r in groups) if closed else 0
            if legal == 1:
                terminal += count
            elif legal > 1:
                independent += count
    return dict(states=states, transitions=branches, terminal_actions_closed=terminal,
                independent_states_closed=independent)


def key_bits(people, genes):
    return 2+(people-2).bit_length()+genes*(17*(people-2)+13).bit_length()
