"""Standalone software checks; synthetic cases are not scientific experiments."""

from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from pedigree_panel_scaling import (
    NuclearInference, PedigreeCase, PedigreeInference, choose_action,
    make_inference, synthetic_case,
)


def small_case():
    genes = ("a", "b")
    return PedigreeCase(
        people=("F", "M", "S", "C", "G"),
        relationships=(("C", "F", "M"), ("G", "C", "S")), genes=genes,
        allele_freqs={"a": 0.2, "b": 0.35},
        a_gene=dict.fromkeys(genes, -0.4), b_gene=dict.fromkeys(genes, -0.1),
        omega_gene=dict.fromkeys(genes, 0.01), delta_gene=dict.fromkeys(genes, 0.5),
        fixed_cost=0.03, variable_cost=0.04,
    )


def enumerate_posterior(case, observed):
    """Independent per-gene enumeration, used only for the five-person check."""
    names = {name: i for i, name in enumerate(case.people)}
    parents = {names[c]: (names[a], names[b]) for c, a, b in case.relationships}
    n = len(case.people)
    marginal = np.zeros((len(case.genes), n, 3))
    pair = np.zeros((len(case.genes), n, n, 3, 3))
    for gene, name in enumerate(case.genes):
        q = case.allele_freqs[name]
        founder = ((1-q)**2, 2*q*(1-q), q*q)
        total = 0.0
        for assignment in product(range(3), repeat=n):
            if any(value >= 0 and assignment[i] != value
                   for i, value in enumerate(observed[gene])):
                continue
            weight = 1.0
            for person, genotype in enumerate(assignment):
                if person not in parents:
                    weight *= founder[genotype]
                else:
                    a, b = (assignment[p]/2 for p in parents[person])
                    weight *= ((1-a)*(1-b), a*(1-b)+(1-a)*b, a*b)[genotype]
            total += weight
            for i, gi in enumerate(assignment):
                marginal[gene, i, gi] += weight
                for j, gj in enumerate(assignment):
                    pair[gene, i, j, gi, gj] += weight
        marginal[gene] /= total
        pair[gene] /= total
    return marginal, pair


class ScalingChecks(unittest.TestCase):
    def assert_normalized(self, state):
        np.testing.assert_allclose(state.genotype.sum(axis=-1), 1, atol=1e-12)
        self.assertTrue(np.all(state.genotype >= 0))
        self.assertTrue(np.all(np.isfinite(state.genotype)))
        self.assertTrue(np.all(np.isinf(state.test_costs[state.tested])))
        self.assertTrue(np.all(np.isneginf(state.test_rewards[state.tested])))
        for person in np.flatnonzero(state.tested):
            np.testing.assert_allclose(state.genotype[:, person],
                np.eye(3)[state.observations[:, person]], atol=1e-12, rtol=0)

    def test_small_posterior_pairs_and_expected_risk(self):
        case = small_case()
        engine = PedigreeInference(case)
        observed = engine.observe(engine.root_state, 3, np.array([1, 2]))
        observed = engine.observe(observed, 4, np.array([0, 1]))
        for history in (engine.root_state, observed):
            state = engine.state(history, with_pairs=True)
            marginal, pair = enumerate_posterior(case, history)
            np.testing.assert_allclose(state.genotype, marginal, atol=1e-12)
            np.testing.assert_allclose(state.pair_genotype, pair, atol=1e-12)
            self.assert_normalized(state)
            for person in np.flatnonzero(~state.tested):
                expected = 0.0
                mass = 0.0
                for panel in product(range(3), repeat=len(case.genes)):
                    probability = np.prod(marginal[np.arange(2), person, panel])
                    if probability > 0:
                        future = engine.observe(history, int(person), np.array(panel))
                        expected += probability * engine.state(future).potential
                        mass += probability
                self.assertAlmostEqual(mass, 1.0, places=12)
                self.assertAlmostEqual(state.myopic_gains[person],
                    state.potential-expected-state.test_costs[person], places=12)

    def test_target_sizes_specialization_and_policy_trajectories(self):
        for family, n, g in product(("nuclear", "multigeneration"), (10, 12, 15), (10, 12, 15)):
            with self.subTest(family=family, people=n, genes=g):
                case = synthetic_case(n, g, family)
                engine = make_inference(case)
                self.assertEqual(engine.graphical_width, 2)
                history = engine.observe(engine.root_state, 2, np.ones(g, dtype=np.int8))
                world = engine.sample_worlds(history, 1, np.random.default_rng(n*100+g))[0]
                self.assertTrue(np.all((world >= 0) & (world <= 2)))
                np.testing.assert_array_equal(world[:, 2], history[:, 2])
                if family == "nuclear":
                    generic = PedigreeInference(case)
                    later = engine.observe(history, 0, world[:, 0])
                    for evidence in (engine.root_state, history, later):
                        actual = engine.state(evidence, with_pairs=True)
                        expected = generic.state(evidence, with_pairs=True)
                        for field in ("genotype", "pair_genotype", "myopic_gains",
                                      "test_costs", "test_rewards"):
                            np.testing.assert_allclose(getattr(actual, field),
                                                       getattr(expected, field), atol=1e-11)
                        np.testing.assert_allclose(engine.rh1_gains_batch(evidence[None])[0],
                                                   expected.myopic_gains, atol=1e-11)
                        for policy in ("recursive_myopic", "information_greedy"):
                            self.assertEqual(choose_action(engine, evidence, policy),
                                             choose_action(generic, evidence, policy))
                for policy in ("recursive_myopic", "information_greedy"):
                    evidence = history.copy()
                    for _ in range(n):
                        state = engine.state(evidence)
                        self.assert_normalized(state)
                        action = choose_action(engine, evidence, policy)
                        if action == -1:
                            break
                        self.assertFalse(state.tested[action])
                        updated = engine.observe(evidence, action, world[:, action])
                        np.testing.assert_array_equal(updated[evidence >= 0], evidence[evidence >= 0])
                        self.assertEqual(np.count_nonzero(updated >= 0),
                                         np.count_nonzero(evidence >= 0)+g)
                        evidence = updated
                    else:
                        self.fail("Policy did not terminate within the legal testing horizon")

    def test_initial_evidence_and_public_state_contract(self):
        case = synthetic_case(10, 10)
        case = replace(case, initial_evidence=(("F", (1,)*10),))
        for kind in (PedigreeInference, NuclearInference):
            engine = kind(case)
            root = engine.root_state
            worlds = engine.sample_worlds(root, 3, np.random.default_rng(7))
            np.testing.assert_array_equal(worlds[:, :, 0], np.ones((3, 10)))
            with self.assertRaisesRegex(ValueError, "once"):
                engine.observe(root, 0, np.ones(10, dtype=int))
            with self.assertRaises(ValueError):
                engine.observe(root, 1, np.zeros(9, dtype=int))
            partial = root.copy()
            partial[0, 1] = 0
            with self.assertRaisesRegex(ValueError, "complete panels"):
                engine.state(partial)
            removed = root.copy()
            removed[:, 0] = -1
            with self.assertRaisesRegex(ValueError, "preserve initial"):
                engine.state(removed)
            with self.assertRaises(ValueError):
                engine.state(root.astype(float))
            self.assertEqual(engine.sample_worlds(root, 0, np.random.default_rng(0)).shape, (0, 10, 10))

    def test_deterministic_support_costs_and_stop_tie(self):
        original = synthetic_case(10, 10)
        for kind in (PedigreeInference, NuclearInference):
            deterministic = replace(original, allele_freqs={g: float(i % 2)
                                    for i, g in enumerate(original.genes)})
            engine = kind(deterministic)
            state = engine.state(engine.root_state, with_pairs=True)
            self.assert_normalized(state)
            self.assertAlmostEqual(state.potential, 0)
            np.testing.assert_allclose(state.test_costs, original.fixed_cost+original.variable_cost)
            contradiction = engine.observe(engine.root_state, 0, np.ones(10, dtype=int))
            with self.assertRaisesRegex(ValueError, "zero support"):
                engine.state(contradiction)
            with self.assertRaisesRegex(ValueError, "zero support"):
                engine.sample_worlds(contradiction, 1, np.random.default_rng(1))
            ordinary = kind(original)
            current = ordinary.state(ordinary.root_state)
            expected_cost = original.fixed_cost+original.variable_cost*(1-np.prod(1-current.carrier, axis=0))
            np.testing.assert_allclose(current.test_costs, expected_cost)
            zero = dict.fromkeys(original.genes, 0.0)
            tied = kind(replace(original, a_gene=zero, b_gene=zero, omega_gene=zero,
                                fixed_cost=0, variable_cost=0))
            for policy in ("recursive_myopic", "information_greedy"):
                self.assertEqual(choose_action(tied, tied.root_state, policy), -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
