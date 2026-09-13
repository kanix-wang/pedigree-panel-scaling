"""Exact current-state pedigree inference without enumerating panel outcomes.

Genes are independent conditional on observed panels. NumPy sum-product batches
genes and, when requested, one-person genotype clamps on the canonical junction
tree. Its largest factor depends on pedigree treewidth, not panel size. The
public observation array has shape (genes, people), with -1 denoting an unknown
genotype; legal public states observe complete panels. No future state or full
pedigree joint is materialized.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .model import PedigreeCase
from .junction_tree import build_canonical_junction_tree

FloatArray = NDArray[np.float64]
ObservationArray = NDArray[np.int8]


@dataclass(frozen=True)
class InferenceState:
    observations: ObservationArray
    tested: NDArray[np.bool_]
    genotype: FloatArray
    carrier: FloatArray
    potential: float
    baseline: float
    stop_reward: float
    test_costs: FloatArray
    test_rewards: FloatArray
    # Axes: gene, target person, tested person, target genotype, test genotype.
    pair_genotype: FloatArray | None = None
    # Axes: gene, target person, tested person, test genotype.
    pair_carrier_genotype: FloatArray | None = None
    myopic_gains: FloatArray | None = None

    @property
    def psi(self) -> float:
        return self.potential

    @property
    def H(self) -> float:
        return self.baseline

    @property
    def p(self) -> FloatArray:
        return self.carrier


class PedigreeInference:
    """Batched exact junction-tree kernel using the production model semantics.

    ``state(..., with_pairs=True)`` provides expected one-step risk reduction
    minus complete test cost. That is the horizon-one information comparator;
    production's recursive-myopic uses ``test_rewards`` versus ``stop_reward``.
    Caches are bounded and contain observed-state quantities only.
    """

    def __init__(self, case: PedigreeCase, *, cache_size: int = 8) -> None:
        self.case = case
        self.people, self.genes = tuple(case.people), tuple(case.genes)
        self.n_people, self.n_genes = len(self.people), len(self.genes)
        if not self.genes or len(set(self.genes)) != self.n_genes:
            raise ValueError("Genes must be nonempty and unique")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self._cache_size = cache_size
        self._cache: OrderedDict[bytes, InferenceState] = OrderedDict()
        self.tree = build_canonical_junction_tree(
            people=self.people, relationships=case.relationships
        )
        self._person_rank = {p: i for i, p in enumerate(self.people)}
        self.moral_edges = tuple(
            (self._person_rank[a], self._person_rank[b])
            for a, b in self.tree.moral_edges
        )
        self.graphical_width = self.tree.treewidth
        self._clique_rank = {c.clique_id: i for i, c in enumerate(self.tree.cliques)}
        self._scopes = [
            tuple(self._person_rank[p] for p in c.members) for c in self.tree.cliques
        ]
        self._homes = [self._clique_rank[self.tree.home_cliques[p]] for p in self.people]
        self._neighbors: list[list[int]] = [[] for _ in self._scopes]
        self._separators: dict[tuple[int, int], tuple[int, ...]] = {}
        for edge in self.tree.edges:
            a, b = self._clique_rank[edge.left], self._clique_rank[edge.right]
            sep = tuple(self._person_rank[p] for p in edge.separator)
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)
            self._separators[a, b] = self._separators[b, a] = sep
        self._parent = {0: -1}
        self._order = [0]
        for node in self._order:
            for neighbor in self._neighbors[node]:
                if neighbor != self._parent[node]:
                    self._parent[neighbor] = node
                    self._order.append(neighbor)

        self._parameters(case)
        self._base = self._make_factors()
        self._root = np.full((self.n_genes, self.n_people), -1, dtype=np.int8)
        seen: set[str] = set()
        for person, panel in case.initial_evidence:
            if person in seen or person not in self._person_rank:
                raise ValueError("Initial evidence has duplicate or unknown person")
            values = np.asarray(panel)
            if values.shape != (self.n_genes,) or values.dtype.kind not in "iu":
                raise ValueError("Initial evidence must contain integer complete panels")
            if np.any((values < 0) | (values > 2)):
                raise ValueError("Initial genotypes must be in 0,1,2")
            self._root[:, self._person_rank[person]] = values
            seen.add(person)
        # Validate root support now, including deterministic allele frequencies.
        self.state(self._root)

    def _parameters(self, case: PedigreeCase) -> None:
        mappings = (case.allele_freqs, case.a_gene, case.b_gene,
                    case.delta_gene, case.omega_gene)
        if any(set(m) != set(self.genes) for m in mappings):
            raise ValueError("Parameter maps must cover exactly the case genes")
        values = [np.array([m[g] for g in self.genes], dtype=float) for m in mappings]
        if not all(np.all(np.isfinite(v)) for v in values):
            raise ValueError("Parameters must be finite")
        self.frequencies = values[0]
        if np.any((self.frequencies < 0) | (self.frequencies > 1)):
            raise ValueError("Allele frequencies must be in [0,1]")
        children = {row[0] for row in case.relationships}
        multiplier = np.array([2.0 if p in children else 1.0 for p in self.people])
        self.a = values[1][:, None] * multiplier[None, :]
        self.b = values[2][:, None] * multiplier[None, :]
        self.delta = np.broadcast_to(values[3][:, None], self.a.shape).copy()
        self.omega = values[4][:, None] * multiplier[None, :]
        self.risk_weights = -(self.a * self.delta + self.b)
        if np.any(self.risk_weights < 0):
            raise ValueError("Bayes-risk weights must be nonnegative")
        if any(not np.isfinite(c) or c < 0 for c in (case.fixed_cost, case.variable_cost)):
            raise ValueError("Test costs must be finite and nonnegative")

    @staticmethod
    def _expand(values: FloatArray, scope: tuple[int, ...], target: tuple[int, ...]) -> FloatArray:
        # Both scopes retain the common pedigree order.
        return values.reshape((*values.shape[:2], *(3 if p in scope else 1 for p in target)))

    def _make_factors(self) -> list[FloatArray]:
        result = [np.ones((1, self.n_genes, *((3,) * len(s)))) for s in self._scopes]
        parents = {c: (p1, p2) for c, p1, p2 in self.case.relationships}
        for factor_id, names in self.tree.factor_scopes.items():
            scope = tuple(self._person_rank[p] for p in names)
            home = self._clique_rank[self.tree.factor_homes[factor_id]]
            if factor_id.startswith("founder:"):
                f = self.frequencies
                table = np.stack(((1 - f) ** 2, 2 * f * (1 - f), f ** 2), axis=-1)[None]
            else:
                child = factor_id.removeprefix("family:")
                parent1, parent2 = parents[child]
                assignments = np.array(list(product(range(3), repeat=3)))
                x = assignments[:, names.index(parent1)] / 2
                y = assignments[:, names.index(parent2)] / 2
                c = assignments[:, names.index(child)]
                probabilities = np.where(c == 0, (1 - x) * (1 - y),
                                         np.where(c == 1, x * (1 - y) + (1 - x) * y, x * y))
                table = np.broadcast_to(probabilities.reshape(1, 1, 3, 3, 3),
                                        (1, self.n_genes, 3, 3, 3))
            result[home] *= self._expand(table, scope, self._scopes[home])
        return result

    @property
    def root_state(self) -> ObservationArray:
        return self._root.copy()

    def _observations(self, observations: ObservationArray) -> ObservationArray:
        raw = np.asarray(observations)
        if raw.shape != (self.n_genes, self.n_people) or raw.dtype.kind not in "iu":
            raise ValueError("Observations must be an integer (genes, people) array")
        if np.any((raw < -1) | (raw > 2)):
            raise ValueError("Observed genotypes must be -1,0,1,2")
        known = raw >= 0
        if np.any(np.any(known, axis=0) != np.all(known, axis=0)):
            raise ValueError("Public states must observe complete panels")
        if np.any((self._root >= 0) & (raw != self._root)):
            raise ValueError("Observations must preserve initial evidence")
        return raw.astype(np.int8, copy=True)

    def _calibrate(self, observations: ObservationArray) -> list[FloatArray]:
        """Return normalized clique beliefs, batching (states, genes).

        Zero-support clamps produce all-zero beliefs and are retained as such;
        public state queries reject them. Message normalization prevents a long
        pedigree's probability scale from underflowing during propagation.
        """
        batch = observations.shape[0]
        local = [np.broadcast_to(v, (batch, *v.shape[1:])).copy() for v in self._base]
        for person, home in enumerate(self._homes):
            obs = observations[:, :, person]
            mask = (obs[:, :, None] < 0) | (obs[:, :, None] == np.arange(3))
            local[home] *= self._expand(mask, (person,), self._scopes[home])
        messages: dict[tuple[int, int], FloatArray] = {}

        def send(source: int, destination: int) -> None:
            value = local[source].copy()
            for neighbor in self._neighbors[source]:
                if neighbor != destination:
                    value *= self._expand(messages[neighbor, source],
                                          self._separators[neighbor, source], self._scopes[source])
            sep = self._separators[source, destination]
            axes = tuple(2 + k for k, p in enumerate(self._scopes[source]) if p not in sep)
            value = value.sum(axis=axes) if axes else value
            norm = value.sum(axis=tuple(range(2, value.ndim)), keepdims=True)
            messages[source, destination] = np.divide(value, norm, out=np.zeros_like(value), where=norm > 0)

        for source in reversed(self._order[1:]):
            send(source, self._parent[source])
        for source in self._order:
            for destination in self._neighbors[source]:
                if destination != self._parent[source]:
                    send(source, destination)
        beliefs = []
        for source, value in enumerate(local):
            for neighbor in self._neighbors[source]:
                value *= self._expand(messages[neighbor, source],
                                      self._separators[neighbor, source], self._scopes[source])
            norm = value.sum(axis=tuple(range(2, value.ndim)), keepdims=True)
            beliefs.append(np.divide(value, norm, out=np.zeros_like(value), where=norm > 0))
        return beliefs

    def _marginals(self, beliefs: list[FloatArray]) -> FloatArray:
        result = np.empty((*beliefs[0].shape[:2], self.n_people, 3))
        for person, home in enumerate(self._homes):
            axes = tuple(2 + k for k, p in enumerate(self._scopes[home]) if p != person)
            result[:, :, person] = beliefs[home].sum(axis=axes) if axes else beliefs[home]
        return result

    def state(self, observations: ObservationArray, *, with_pairs: bool = False) -> InferenceState:
        observed = self._observations(observations)
        key = observed.tobytes()
        cached = self._cache.get(key)
        if cached is not None and (not with_pairs or cached.pair_genotype is not None):
            self._cache.move_to_end(key)
            return cached
        genotype = (cached.genotype if cached is not None else
                    self._marginals(self._calibrate(observed[None]))[0])
        if not np.allclose(genotype.sum(axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("Evidence has zero support under the pedigree model")
        carrier = np.clip(genotype[:, :, 1:].sum(axis=-1), 0, 1)
        tested = np.all(observed >= 0, axis=0)
        risk = self.risk_weights * carrier * (1 - carrier)
        potential = float(risk[:, ~tested].sum())
        affine = self.a * (1 - self.delta) * carrier + self.omega
        baseline = float(affine[:, ~tested].sum())
        costs = self.case.fixed_cost + self.case.variable_cost * (1 - np.prod(1 - carrier, axis=0))
        rewards = affine.sum(axis=0) - costs
        costs[tested], rewards[tested] = np.inf, -np.inf
        pairs = carrier_pairs = gains = None
        if with_pairs:
            # A single batch carries all 3*N one-person clamps, independently
            # within each gene. Impossible genotype clamps remain zero rows.
            clamped = np.broadcast_to(observed, (self.n_people * 3, *observed.shape)).copy()
            for person in range(self.n_people):
                for g in range(3):
                    row = person * 3 + g
                    # An already observed person's contradictory clamps receive
                    # zero probability when multiplied by its base marginal.
                    clamped[row, :, person] = g
            conditional = self._marginals(self._calibrate(clamped))
            conditional = conditional.reshape(self.n_people, 3, self.n_genes, self.n_people, 3)
            pairs = conditional.transpose(2, 3, 0, 4, 1) * genotype[:, None, :, None, :]
            carrier_pairs = pairs[:, :, :, 1:, :].sum(axis=3)
            expected_square = np.divide(carrier_pairs ** 2, genotype[:, None, :, :],
                                        out=np.zeros_like(carrier_pairs),
                                        where=genotype[:, None, :, :] > 0).sum(axis=-1)
            explained = np.maximum(0, expected_square - carrier[:, :, None] ** 2)
            gains = (explained[:, ~tested, :] * self.risk_weights[:, ~tested, None]).sum(axis=(0, 1)) - costs
            gains[tested] = -np.inf
        result = InferenceState(observed, tested, genotype, carrier, potential,
                                     baseline, baseline - potential, costs, rewards,
                                     pairs, carrier_pairs, gains)
        # Immutable returned arrays prevent callers from silently corrupting a
        # cache entry. Observation updates always create a new array.
        for field in result.__dataclass_fields__:
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        if self._cache_size:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def observe(self, observations: ObservationArray, person: int, panel: ObservationArray) -> ObservationArray:
        observed = self._observations(observations)
        if isinstance(person, bool) or not isinstance(person, (int, np.integer)) or not 0 <= person < self.n_people:
            raise ValueError("Test person must be a valid index")
        if np.any(observed[:, person] >= 0):
            raise ValueError("A person may only be tested once")
        values = np.asarray(panel)
        if values.shape != (self.n_genes,) or values.dtype.kind not in "iu" or np.any((values < 0) | (values > 2)):
            raise ValueError("Panel must contain one integer genotype per gene")
        observed[:, person] = values
        return observed

    def independent_remaining(self, observations: ObservationArray) -> bool:
        observed = self._observations(observations)
        known = np.all(observed >= 0, axis=0)
        return all(known[a] or known[b] for a, b in self.moral_edges)

    def sample_worlds(self, observations: ObservationArray, count: int,
                      rng: np.random.Generator) -> ObservationArray:
        """Sample coherent complete latent worlds conditional on public evidence.

        Sample a root clique, then each clique given its already sampled
        separator. A shared latent world can subsequently supply all panel
        outcomes along an adaptive trajectory without observing hidden values.
        """
        observed = self._observations(observations)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        beliefs = self._calibrate(observed[None])
        if not np.allclose(beliefs[0].reshape(self.n_genes, -1).sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("Evidence has zero support under the pedigree model")
        worlds = np.full((count, self.n_genes, self.n_people), -1, dtype=np.int8)
        for node in self._order:
            scope = self._scopes[node]
            assignments = np.array(list(product(range(3), repeat=len(scope))), dtype=np.int8)
            probabilities = np.broadcast_to(beliefs[node].reshape(self.n_genes, -1),
                                            (count, self.n_genes, len(assignments))).copy()
            parent = self._parent[node]
            if parent >= 0:
                for person in self._separators[node, parent]:
                    probabilities *= (worlds[:, :, person, None] == assignments[None, None, :, scope.index(person)])
            mass = probabilities.sum(axis=-1, keepdims=True)
            if np.any(mass <= 0):
                raise RuntimeError("Sampled separator has zero conditional mass")
            probabilities /= mass
            cdf = probabilities.cumsum(axis=-1)
            cdf[:, :, -1] = 1.0
            selected = (rng.random((count, self.n_genes, 1)) >= cdf).sum(axis=-1)
            worlds[:, :, list(scope)] = assignments[selected]
        return worlds

    def clear_cache(self) -> None:
        self._cache.clear()


__all__ = ["InferenceState", "PedigreeInference"]
