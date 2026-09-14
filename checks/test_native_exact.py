"""Parity of optional compiled recursion, Python recursion, and tiny-world oracle."""

from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from pedigree_panel_scaling import ExactSolution, NuclearInference, solve_exact
from pedigree_panel_scaling._native_exact import available
from pedigree_panel_scaling._nuclear_exact import _solve_python
from test_exact import ExplicitWorldOracle, canonical_case


class OptionalNativeChecks(unittest.TestCase):
    def test_unavailable_extension_keeps_exact_python_available(self):
        engine = NuclearInference(canonical_case())
        expected = _solve_python(engine)
        with patch("pedigree_panel_scaling._native_exact.available", return_value=False), \
             patch("pedigree_panel_scaling._native_exact.solve_native",
                   side_effect=AssertionError("Unavailable extension was called")):
            actual = solve_exact(engine)
        self.assertIs(type(actual), ExactSolution)
        self.assertEqual(actual.root_value, expected.root_value)
        self.assertEqual(dict(actual.root_action_values), dict(expected.root_action_values))
        self.assertEqual(actual.states_evaluated, expected.states_evaluated)

    def test_unloadable_extension_keeps_exact_python_available(self):
        engine = NuclearInference(canonical_case())
        expected = _solve_python(engine)
        with patch("pedigree_panel_scaling._native_exact.available", return_value=True), \
             patch("pedigree_panel_scaling._native_exact.importlib.import_module",
                   side_effect=ImportError("optional binary cannot be loaded")):
            actual = solve_exact(engine)
        self.assertIs(type(actual), ExactSolution)
        self.assertEqual(actual.root_value, expected.root_value)
        self.assertEqual(dict(actual.root_action_values), dict(expected.root_action_values))


@unittest.skipUnless(available(), "Optional compiled extension is not installed")
class NativeExactChecks(unittest.TestCase):
    def native_solution(self, engine, observations=None):
        with patch("pedigree_panel_scaling._native_exact._solve_python",
                   side_effect=AssertionError("Representable native case fell back to Python")):
            return solve_exact(engine, observations)

    def test_all_two_gene_histories_match_python_and_independent_oracle(self):
        supported_count = impossible_count = 0
        for frequencies in ((.12, .12), (0.0, .12)):
            base = canonical_case(siblings=True)
            case = replace(base, allele_freqs=dict(zip(base.genes, frequencies)),
                           a_gene=dict.fromkeys(base.genes, -.08), b_gene=dict.fromkeys(base.genes, -.04),
                           delta_gene=dict.fromkeys(base.genes, .6), omega_gene=dict.fromkeys(base.genes, 0.0))
            engine = NuclearInference(case)
            native = self.native_solution(engine)
            python = _solve_python(engine)
            oracle = ExplicitWorldOracle(case)
            expected, root_scores = oracle.solve((-1,)*4)
            self.assertIs(type(native), ExactSolution)
            for solution in (native, python):
                self.assertAlmostEqual(solution.root_value, expected, places=12)
                self.assertEqual(set(solution.root_action_values), set(root_scores))
                for action, value in root_scores.items():
                    self.assertAlmostEqual(solution.root_action_values[action], value, places=12)
            self.assertEqual(native.states_evaluated, python.states_evaluated)
            self.assertEqual(native.transitions_evaluated, python.transitions_evaluated)
            for key in product(range(-1, 9), repeat=4):
                mask = np.ones(len(oracle.worlds), dtype=bool)
                for person, code in enumerate(key):
                    if code >= 0:
                        mask &= oracle.codes[:, person] == code
                observed = np.array([[-1 if code < 0 else code // 3**gene % 3
                                      for code in key] for gene in range(2)], dtype=np.int8)
                if not mask.any():
                    for solution in (native, python):
                        with self.assertRaises(ValueError):
                            solution.value_at(observed)
                    impossible_count += 1
                    continue
                value, scores = oracle.solve(key)
                for solution in (native, python):
                    self.assertAlmostEqual(solution.value_at(observed), value, places=12)
                    self.assertAlmostEqual(scores[solution.action_at(observed)], value, places=12)
                supported_count += 1
            keys = list(native._values)
            self.assertEqual(len(keys), native.states_evaluated)
            self.assertEqual(len(set(keys)), len(keys))
            self.assertEqual(set(native._actions), set(keys))
            self.assertTrue(all(np.isfinite(native._values[key]) for key in keys))
            for mapping in (native.root_action_values, native._values, native._actions):
                with self.assertRaises(TypeError):
                    mapping[next(iter(mapping))] = 0
            root_key = native._codec.key(engine.root_state)
            with self.assertRaises(KeyError):
                native._values[(8, 0, *root_key[2:])]
            with self.assertRaises(KeyError):
                native._values[(3, 0, *root_key[2:])]
        self.assertEqual(supported_count, 2416)
        self.assertEqual(impossible_count, 17584)

    def test_custom_root_and_uncached_terminal_queries(self):
        case = replace(canonical_case(siblings=True), initial_evidence=(("C", (1, 0)),))
        engine = NuclearInference(case)
        root = engine.observe(engine.root_state, 0, np.array([1, 0]))
        native = self.native_solution(engine, root)
        python = _solve_python(engine, root)
        self.assertAlmostEqual(native.root_value, python.root_value, places=12)
        self.assertEqual(native.action_at(), native.root_action)
        world = engine.sample_worlds(root, 1, np.random.default_rng(42))[0]
        for person in (1, 3):
            partial = engine.observe(root, person, world[:, person])
            self.assertAlmostEqual(native.value_at(partial), python.value_at(partial), places=12)
        self.assertEqual(native.value_at(world), 0.0)
        self.assertEqual(native.action_at(world), -1)
        with self.assertRaises(ValueError):
            native.value_at(engine.root_state)
        invalid = root.copy()
        invalid[0, 1] = 0
        with self.assertRaises(ValueError):
            native.action_at(invalid)

    def test_unrepresentable_keys_fall_back_without_changing_the_problem(self):
        base = canonical_case()
        genes = tuple(f"g{i}" for i in range(15))
        case = replace(base, genes=genes, allele_freqs=dict.fromkeys(genes, .1),
                       a_gene=dict.fromkeys(genes, -.08), b_gene=dict.fromkeys(genes, -.04),
                       delta_gene=dict.fromkeys(genes, .6), omega_gene=dict.fromkeys(genes, 0.0),
                       initial_evidence=tuple((person, (0,)*15) for person in base.people))
        engine = NuclearInference(case)
        with patch("pedigree_panel_scaling._native_exact._solve_python", wraps=_solve_python) as fallback:
            solution = solve_exact(engine)
        self.assertTrue(fallback.called)
        self.assertEqual(solution.root_value, 0.0)
        self.assertEqual(solution.root_action, -1)
        self.assertEqual(solution.states_evaluated, 1)
        self.assertEqual(solution.transitions_evaluated, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
