"""Exact nuclear-family specialization of the pedigree inference kernel.

Observed siblings enter the nine-state parental posterior through counts of
their three genotypes. This is a computational specialization of standard
pedigree factorization, not a new inference theorem. Public model, evidence,
test-once, and reward semantics are inherited unchanged from the generic engine.
"""

from __future__ import annotations

import numpy as np

from .model import PedigreeCase
from .inference import (
    FloatArray,
    ObservationArray,
    PedigreeInference,
)


def is_single_nuclear_family(case: PedigreeCase) -> bool:
    """Require exactly two founders and all other people to be their children."""
    if not case.relationships:
        return False
    pairs = {frozenset((a, b)) for _, a, b in case.relationships}
    if len(pairs) != 1:
        return False
    parents = next(iter(pairs))
    children = [c for c, _, _ in case.relationships]
    return (len(parents) == 2 and len(set(children)) == len(children)
            and not parents.intersection(children)
            and set(case.people) == parents.union(children))


class NuclearInference(PedigreeInference):
    """Use a nine-cell parental posterior and shared sibling likelihoods.

    The private calibration result contains all marginals directly, avoiding
    a separate clique pass. Inherited ``state`` still computes exactly the same
    complete costs, rewards and optional pair distributions using batched clamps.
    """

    def __init__(self, case: PedigreeCase, *, cache_size: int = 8) -> None:
        if not is_single_nuclear_family(case):
            raise ValueError("NuclearInference requires a single parental pair")
        parent_names = set(case.relationships[0][1:])
        self.parent_indices = tuple(i for i, p in enumerate(case.people) if p in parent_names)
        self.child_indices = tuple(i for i, p in enumerate(case.people) if p not in parent_names)
        self._parent_genotypes = np.array([(a, b) for a in range(3) for b in range(3)], dtype=np.int8)
        x, y = self._parent_genotypes[:, 0] / 2, self._parent_genotypes[:, 1] / 2
        self._child_cpd = np.stack(((1 - x) * (1 - y), x * (1 - y) + (1 - x) * y, x * y), axis=-1)
        # Base initialization validates the model and calls self.state(root),
        # hence the specialization's structural constants are established first.
        super().__init__(case, cache_size=cache_size)

    def _parent_posterior(self, observations: ObservationArray) -> FloatArray:
        frequency = self.frequencies
        founder = np.stack(((1 - frequency) ** 2, 2 * frequency * (1 - frequency), frequency ** 2), axis=-1)
        prior = (founder[:, self._parent_genotypes[:, 0]]
                 * founder[:, self._parent_genotypes[:, 1]])
        children = observations[:, :, self.child_indices]
        counts = (children[:, :, :, None] == np.arange(3)).sum(axis=2)
        likelihood = np.prod(self._child_cpd[None, None, :, :] ** counts[:, :, None, :], axis=-1)
        posterior = likelihood * prior[None]
        for k, parent in enumerate(self.parent_indices):
            observed = observations[:, :, parent, None]
            posterior *= (observed < 0) | (observed == self._parent_genotypes[None, None, :, k])
        normalizer = posterior.sum(axis=-1, keepdims=True)
        return np.divide(posterior, normalizer, out=np.zeros_like(posterior), where=normalizer > 0)

    def _calibrate(self, observations: ObservationArray) -> list[FloatArray]:
        posterior = self._parent_posterior(observations)
        child_marginal = posterior @ self._child_cpd
        genotype = np.broadcast_to(child_marginal[:, :, None, :],
                                  (*posterior.shape[:2], self.n_people, 3)).copy()
        # Observed children have point masses, conditional on nonzero support.
        known = observations >= 0
        support = posterior.sum(axis=-1)
        genotype = np.where(known[:, :, :, None],
                            (observations[:, :, :, None] == np.arange(3)) * support[:, :, None, None],
                            genotype)
        parental_grid = posterior.reshape(*posterior.shape[:2], 3, 3)
        genotype[:, :, self.parent_indices[0]] = parental_grid.sum(axis=-1)
        genotype[:, :, self.parent_indices[1]] = parental_grid.sum(axis=-2)
        return [genotype]

    def _marginals(self, beliefs: list[FloatArray]) -> FloatArray:
        return beliefs[0]

    def rh1_gains_batch(self, observations: ObservationArray,
                       costs: FloatArray | None = None) -> FloatArray:
        """Exact one-test expected risk reduction minus panel cost, batch x people.

        Unobserved siblings share their conditional genotype law given the nine
        parental states. Their information contribution can therefore be
        weighted and summed without a people-by-people pair tensor. Distinct
        siblings use a conditional-independence moment, while the tested
        person's own genotype observation removes its entire current variance.
        ``costs``, when supplied, must be current complete expected panel costs
        for these same public histories (as in the batched continuation scorer).
        """
        observed = np.asarray(observations)
        if (observed.ndim != 3 or observed.shape[1:] != (self.n_genes, self.n_people)
                or observed.dtype.kind not in "iu" or np.any((observed < -1) | (observed > 2))):
            raise ValueError("RH1 histories must be an integer batch of public genotype arrays")
        known = observed >= 0
        tested = known.all(axis=1)
        if np.any(known.any(axis=1) != tested):
            raise ValueError("RH1 public histories must observe complete panels")
        if np.any((self._root[None] >= 0) & (observed != self._root[None])):
            raise ValueError("RH1 histories must preserve initial evidence")
        posterior = self._parent_posterior(observed)
        if not np.allclose(posterior.sum(axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("RH1 planning history has zero support")
        grid = posterior.reshape(*posterior.shape[:2], 3, 3)
        parent_genotype = (grid.sum(axis=-1), grid.sum(axis=-2))
        parent_carrier = tuple(p[..., 1:].sum(axis=-1) for p in parent_genotype)
        child_genotype = posterior @ self._child_cpd
        child_carrier = child_genotype[..., 1:].sum(axis=-1)
        conditional_child_carrier = self._child_cpd[:, 1:].sum(axis=-1)
        child_weight = self.risk_weights[:, self.child_indices]
        child_weight_sum = (child_weight[None] * ~tested[:, None, self.child_indices]).sum(axis=-1)

        def explained(joint, genotype, carrier):
            second = np.divide(joint*joint, genotype, out=np.zeros_like(joint),
                               where=genotype > 0).sum(axis=-1)
            return np.maximum(0, second-carrier*carrier)

        by_gene = np.zeros((*posterior.shape[:2], self.n_people))
        parent_to_child_information = []
        for k, parent in enumerate(self.parent_indices):
            other = self.parent_indices[1-k]
            mask = (self._parent_genotypes[:, k, None] == np.arange(3)).astype(float)
            other_carrier = (self._parent_genotypes[:, 1-k] > 0).astype(float)
            child_joint = (posterior * conditional_child_carrier) @ mask
            other_joint = (posterior * other_carrier) @ mask
            own_risk = self.risk_weights[:, parent] * parent_carrier[k] * (1-parent_carrier[k])
            other_information = explained(other_joint, parent_genotype[k], parent_carrier[1-k])
            child_information = explained(child_joint, parent_genotype[k], child_carrier)
            by_gene[:, :, parent] = (
                own_risk + self.risk_weights[:, other] * ~tested[:, None, other] * other_information
                + child_weight_sum * child_information)
            parent_child_joint = (posterior * (self._parent_genotypes[:, k] > 0)) @ self._child_cpd
            parent_to_child_information.append(
                self.risk_weights[:, parent] * ~tested[:, None, parent]
                * explained(parent_child_joint, child_genotype, parent_carrier[k]))
        sibling_joint = (posterior * conditional_child_carrier) @ self._child_cpd
        sibling_information = explained(sibling_joint, child_genotype, child_carrier)
        other_weights = child_weight_sum[:, :, None] - child_weight[None] * ~tested[:, None, self.child_indices]
        by_gene[:, :, self.child_indices] = (
            child_weight[None] * (child_carrier*(1-child_carrier))[:, :, None]
            + other_weights * sibling_information[:, :, None]
            + sum(parent_to_child_information)[:, :, None])
        if costs is None:
            negative = np.broadcast_to(child_genotype[:, :, None, 0], by_gene.shape).copy()
            for k, parent in enumerate(self.parent_indices):
                negative[:, :, parent] = parent_genotype[k][:, :, 0]
            costs = self.case.fixed_cost + self.case.variable_cost * (1-negative.prod(axis=1))
            costs[tested] = np.inf
        else:
            costs = np.asarray(costs)
            if costs.shape != tested.shape or not np.isfinite(costs[~tested]).all() or np.any(costs[~tested] < 0):
                raise ValueError("RH1 costs must be current nonnegative complete-panel costs")
        gains = by_gene.sum(axis=1)-costs
        gains[tested] = -np.inf
        return gains

    def sample_worlds(self, observations: ObservationArray, count: int,
                      rng: np.random.Generator) -> ObservationArray:
        """Exact posterior worlds; random stream differs from generic sampling."""
        observed = self._observations(observations)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Sample count must be a nonnegative integer")
        posterior = self._parent_posterior(observed[None])[0]
        if not np.allclose(posterior.sum(axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("Evidence has zero support under the pedigree model")
        parent_cdf = posterior.cumsum(axis=-1)
        parent_cdf[:, -1] = 1.0
        choices = (rng.random((count, self.n_genes, 1)) >= parent_cdf[None]).sum(axis=-1)
        worlds = np.broadcast_to(observed, (count, *observed.shape)).copy()
        worlds[:, :, list(self.parent_indices)] = self._parent_genotypes[choices]
        # Given the sampled parental pair, unobserved siblings are independent.
        child_cdf = self._child_cpd[choices].cumsum(axis=-1)
        child_cdf[:, :, -1] = 1.0
        draws = (rng.random((count, self.n_genes, len(self.child_indices), 1))
                 >= child_cdf[:, :, None, :]).sum(axis=-1)
        child_evidence = observed[:, self.child_indices]
        worlds[:, :, list(self.child_indices)] = np.where(child_evidence[None] >= 0,
                                                          child_evidence[None], draws)
        return worlds.astype(np.int8, copy=False)


def make_inference(case: PedigreeCase, *, cache_size: int = 8) -> PedigreeInference:
    """Select the exact specialization only on its verified structural domain."""
    engine_type = NuclearInference if is_single_nuclear_family(case) else PedigreeInference
    return engine_type(case, cache_size=cache_size)


__all__ = ["NuclearInference", "is_single_nuclear_family", "make_inference"]
