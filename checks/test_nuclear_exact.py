"""Integration checks for the exact nuclear backend and guarded fallback."""

from dataclasses import FrozenInstanceError, replace
from itertools import product
import math
import unittest
from unittest.mock import patch

import numpy as np

from pedigree_panel_scaling import ExactSolution, NuclearInference, PedigreeInference, solve_exact
from pedigree_panel_scaling._nuclear_exact import supports_nuclear
from test_exact import ExplicitWorldOracle, canonical_case


class NuclearExactChecks(unittest.TestCase):
    def test_partial_root_queries_and_readonly_unified_solution(self):
        case = replace(canonical_case(siblings=True), initial_evidence=(("C", (1, 0)),))
        engine = NuclearInference(case)
        root = engine.observe(engine.root_state, 0, np.array([1, 0]))
        solution = solve_exact(engine, root)
        self.assertIs(type(solution), ExactSolution)
        self.assertGreater(solution.locus_states, 0)
        oracle = ExplicitWorldOracle(case)
        expected, root_scores = oracle.solve(oracle.history_key(root))
        self.assertAlmostEqual(solution.root_value, expected, places=12)
        for action, value in root_scores.items():
            self.assertAlmostEqual(solution.root_action_values[action], value, places=12)
        for panels in product(range(9), repeat=2):
            observed = root.copy()
            for person, code in zip((1, 3), panels):
                observed[:, person] = (code % 3, code // 3)
            key = oracle.history_key(observed)
            supported = np.all(oracle.codes == np.array(key)[None], axis=1).any()
            if not supported:
                with self.assertRaises(ValueError):
                    solution.value_at(observed)
            else:
                self.assertEqual(solution.value_at(observed), 0)
                self.assertEqual(solution.action_at(observed), -1)
        child_observed = engine.observe(root, 3, np.array([0, 0]))
        value, scores = oracle.solve(oracle.history_key(child_observed))
        self.assertAlmostEqual(solution.value_at(child_observed), value, places=12)
        self.assertAlmostEqual(scores[solution.action_at(child_observed)], value, places=12)
        with self.assertRaises(ValueError):
            solution.value_at(engine.root_state)
        with self.assertRaises(ValueError):
            solution._root[0, 0] = 0
        for mapping in (solution.root_action_values, solution._values,
                        solution._actions, solution._codec.memo):
            with self.assertRaises(TypeError):
                mapping[next(iter(mapping))] = 0
        with self.assertRaises(FrozenInstanceError):
            solution.root_value = 0
        root[:, 0] = 2
        self.assertAlmostEqual(solution.value_at(), expected, places=12)

    def test_closed_sibling_ties_and_later_action_identity(self):
        case = replace(canonical_case(siblings=True),
                       initial_evidence=(("F", (1, 1)), ("M", (0, 0))))
        engine = NuclearInference(case)
        solution = solve_exact(engine)
        self.assertEqual(solution.root_action, 2)
        self.assertEqual(solution.root_action_values[2], solution.root_action_values[3])
        next_state = engine.observe(engine.root_state, 2, np.array([0, 1]))
        self.assertEqual(solution.action_at(next_state), 3)
        renamed = engine.observe(engine.root_state, 3, np.array([0, 1]))
        self.assertEqual(solution.action_at(renamed), 2)
        self.assertEqual(solution.value_at(next_state), solution.value_at(renamed))
        complete = engine.observe(next_state, 3, np.array([1, 0]))
        self.assertEqual(solution.action_at(complete), -1)

    def test_modified_kernel_falls_back_to_full_observation_solver(self):
        engine = NuclearInference(canonical_case())
        # A valid alternate conditional law makes the child independent of its
        # parents, giving independent analytic action values for this check.
        engine._child_cpd = np.tile((0.25, 0.5, 0.25), (9, 1))
        engine.clear_cache()
        self.assertFalse(supports_nuclear(engine))
        with patch("pedigree_panel_scaling._nuclear_exact.solve_nuclear",
                   side_effect=AssertionError("unsupported kernel entered backend")):
            solution = solve_exact(engine)
        self.assertIs(type(solution), ExactSolution)
        self.assertIsNone(solution.locus_states)
        p = np.column_stack((1-(1-engine.frequencies)**2,
                             1-(1-engine.frequencies)**2, np.full(2, 0.75)))
        affine = engine.a*(1-engine.delta)*p+engine.omega
        risk = -(engine.a*engine.delta+engine.b)*p*(1-p)
        stop = (affine-risk).sum(axis=0)
        test = affine.sum(axis=0)-engine.case.fixed_cost-engine.case.variable_cost*(1-np.prod(1-p, axis=0))
        expected = {-1: float(stop.sum())}
        for person in range(3):
            expected[person] = float(test[person]+math.fsum(max(stop[j],test[j])
                                                          for j in range(3) if j != person))
        for person, value in expected.items():
            self.assertAlmostEqual(solution.root_action_values[person], value, places=12)
        self.assertAlmostEqual(solution.root_value, max(expected.values()), places=12)
        permuted = NuclearInference(canonical_case())
        permuted._parent_genotypes = permuted._parent_genotypes[::-1].copy()
        self.assertFalse(supports_nuclear(permuted))
        self.assertFalse(supports_nuclear(PedigreeInference(canonical_case())))

    def test_profiles_with_single_genes_preserve_independent_oracle_values(self):
        case = canonical_case()
        engine = NuclearInference(case)
        solution = solve_exact(engine)
        self.assertTrue(all(stop-start == 1 for start,stop in solution._codec.spans))
        value, scores = ExplicitWorldOracle(case).solve((-1,)*3)
        self.assertAlmostEqual(solution.root_value, value, places=12)
        for action, expected in scores.items():
            self.assertAlmostEqual(solution.root_action_values[action], expected, places=12)

    def test_compact_transitions_match_observed_panel_probabilities(self):
        from pedigree_panel_scaling._nuclear_exact import _FastModel
        base = canonical_case(siblings=True)
        genes = ("x", "y", "z")
        case = replace(base, genes=genes, allele_freqs={"x": .1, "y": .1, "z": .2},
                       a_gene=dict.fromkeys(genes, -.08), b_gene=dict.fromkeys(genes, -.04),
                       delta_gene=dict.fromkeys(genes, .6), omega_gene=dict.fromkeys(genes, 0.0))
        engine = NuclearInference(case)
        observed = engine.observe(engine.root_state, 2, np.array([0, 1, 0]))
        model = _FastModel(engine)
        key = model.key(observed)
        state = engine.state(observed)
        for role, person in ((0, 0), (1, 1), (2, 3)):
            compact, direct = {}, {}
            alternatives = [profile.successors(identifier, role)
                            for profile, identifier in zip(model.flat_profiles, key[2:])]
            for outcomes in product(*alternatives):
                target = model.canonical(key[0] | (1 << role) if role < 2 else key[0],
                                         key[1]-int(role == 2), tuple(row[1] for row in outcomes))
                compact[target] = compact.get(target, 0.0)+math.prod(row[0] for row in outcomes)
            for panel in product(range(3), repeat=3):
                probability = math.prod(state.genotype[g, person, value] for g,value in enumerate(panel))
                if probability:
                    target = model.key(engine.observe(observed, person, np.array(panel)))
                    direct[target] = direct.get(target, 0.0)+probability
            self.assertEqual(set(compact), set(direct))
            self.assertAlmostEqual(math.fsum(compact.values()), 1.0, places=12)
            for target in direct:
                self.assertAlmostEqual(compact[target], direct[target], places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
