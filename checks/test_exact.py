"""Exact-optimizer software checks with an independent tiny-family oracle."""

from dataclasses import replace
from functools import lru_cache
from itertools import product
import unittest

import numpy as np

from pedigree_panel_scaling import NuclearInference, PedigreeCase, PedigreeInference
from pedigree_panel_scaling.exact import ExactLimitExceeded, ExactLimits, solve_exact


def canonical_case(siblings=False):
    people = ("F", "M", "C", "D") if siblings else ("F", "M", "C")
    return PedigreeCase(
        people=people, relationships=tuple((p, "F", "M") for p in people[2:]),
        genes=("a", "b"), allele_freqs={"a": 0.05, "b": 0.08},
        a_gene={"a": -0.08, "b": -0.06}, b_gene={"a": -0.04, "b": -0.03},
        omega_gene={"a": 0.0, "b": 0.0}, delta_gene={"a": 0.6, "b": 0.7},
        fixed_cost=0.01, variable_cost=0.02,
    )


class ExplicitWorldOracle:
    """Enumerate complete worlds and backward decisions without engine methods."""

    def __init__(self, case):
        self.case = case
        self.n, self.g = len(case.people), len(case.genes)
        rank = {name: i for i, name in enumerate(case.people)}
        parents = {rank[c]: (rank[a], rank[b]) for c, a, b in case.relationships}
        worlds, weights = [], []
        for flat in product(range(3), repeat=self.n*self.g):
            world = np.array(flat).reshape(self.g, self.n)
            probability = 1.0
            for gene, name in enumerate(case.genes):
                q = case.allele_freqs[name]
                for person in range(self.n):
                    genotype = world[gene, person]
                    if person in parents:
                        a, b = (world[gene, p]/2 for p in parents[person])
                        probability *= ((1-a)*(1-b), a*(1-b)+(1-a)*b, a*b)[genotype]
                    else:
                        probability *= ((1-q)**2, 2*q*(1-q), q*q)[genotype]
            if probability:
                worlds.append(world)
                weights.append(probability)
        self.worlds, self.weights = np.array(worlds), np.array(weights)
        self.codes = (self.worlds * 3**np.arange(self.g)[None, :, None]).sum(axis=1)
        self.multiplier = np.array([2 if p in parents else 1 for p in range(self.n)])

    def history_key(self, observed):
        return tuple(-1 if np.any(observed[:, p] < 0)
                     else int(observed[:, p] @ (3**np.arange(self.g)))
                     for p in range(self.n))

    @lru_cache(None)
    def solve(self, key):
        support = np.ones(len(self.worlds), dtype=bool)
        for person, code in enumerate(key):
            if code >= 0:
                support &= self.codes[:, person] == code
        probability = self.weights[support]
        probability = probability / probability.sum()
        carrier = np.einsum("w,wgp->gp", probability, self.worlds[support] > 0)
        affine, risk = np.zeros_like(carrier), np.zeros_like(carrier)
        for gene, name in enumerate(self.case.genes):
            a = self.case.a_gene[name]*self.multiplier
            b = self.case.b_gene[name]*self.multiplier
            omega = self.case.omega_gene[name]*self.multiplier
            delta = self.case.delta_gene[name]
            affine[gene] = a*(1-delta)*carrier[gene]+omega
            risk[gene] = -(a*delta+b)*carrier[gene]*(1-carrier[gene])
        untested = [p for p, code in enumerate(key) if code < 0]
        actions = {-1: float((affine-risk)[:, untested].sum())}
        for person in untested:
            expected = 0.0
            cost = 0.0
            for code in np.unique(self.codes[support, person]):
                mass = probability[self.codes[support, person] == code].sum()
                future = list(key)
                future[person] = int(code)
                expected += mass*self.solve(tuple(future))[0]
                cost += mass*(self.case.fixed_cost+self.case.variable_cost*(code != 0))
            actions[person] = float(affine[:, person].sum()-cost+expected)
        return max(actions.values()), actions


class ExactChecks(unittest.TestCase):
    def test_trio_matches_explicit_world_backward_enumeration(self):
        case = canonical_case()
        oracle = ExplicitWorldOracle(case)
        self.assertAlmostEqual(oracle.weights.sum(), 1.0, places=14)
        expected, actions = oracle.solve((-1,)*3)
        self.assertAlmostEqual(expected, -0.05980672625419552, places=13)
        for kind in (PedigreeInference, NuclearInference):
            engine = kind(case)
            solution = solve_exact(engine)
            self.assertAlmostEqual(solution.root_value, expected, places=12)
            self.assertEqual(solution.root_action, 2)
            self.assertEqual(set(solution.root_action_values), set(actions))
            for action, value in actions.items():
                self.assertAlmostEqual(solution.root_action_values[action], value, places=12)
            self.assertEqual(solution.value_at(), solution.root_value)
            self.assertEqual(solution.action_at(), solution.root_action)
            with self.assertRaises(TypeError):
                solution.root_action_values[-1] = 0.0
            history = engine.observe(engine.root_state, 2, np.array([1, 0]))
            value, scores = oracle.solve(oracle.history_key(history))
            self.assertAlmostEqual(solution.value_at(history), value, places=12)
            self.assertAlmostEqual(scores[solution.action_at(history)], value, places=12)
            self.assertGreater(solution.states_evaluated, 1)
            self.assertGreater(solution.transitions_evaluated, 0)
            self.assertGreater(solution.terminal_actions_closed, 0)
            self.assertGreaterEqual(solution.elapsed_seconds, 0)

    def test_sibling_canonical_value_and_parent_tie(self):
        solution = solve_exact(NuclearInference(canonical_case(siblings=True)))
        self.assertAlmostEqual(solution.root_value, -0.08007587340660761, places=12)
        self.assertEqual(solution.root_action, 0)
        self.assertAlmostEqual(solution.root_action_values[0], solution.root_action_values[1], places=12)

    def test_conditional_initial_evidence_and_last_person_closure(self):
        case = replace(canonical_case(), initial_evidence=(("C", (1, 0)),))
        engine = NuclearInference(case)
        root = engine.root_state
        expected, _ = ExplicitWorldOracle(case).solve((-1, -1, 1))
        solution = solve_exact(engine)
        self.assertAlmostEqual(solution.root_value, expected, places=12)
        self.assertNotIn(2, solution.root_action_values)
        history = engine.observe(root, 0, np.array([1, 0]))
        state = engine.state(history)
        final = solve_exact(engine, history)
        self.assertAlmostEqual(final.root_value, max(state.stop_reward, state.test_rewards[1]), places=12)
        self.assertEqual(final.states_evaluated, 1)
        self.assertEqual(final.transitions_evaluated, 0)
        self.assertEqual(final.terminal_actions_closed, 1)
        self.assertEqual(set(final.root_action_values), {-1, 1})
        with self.assertRaises(ValueError):
            final.value_at(root)
        complete = engine.observe(history, 1, np.array([0, 0]))
        self.assertEqual(final.value_at(complete), 0.0)
        self.assertEqual(final.action_at(complete), -1)
        all_tested = solve_exact(engine, complete)
        self.assertEqual(all_tested.root_value, 0.0)
        self.assertEqual(all_tested.root_action, -1)
        self.assertEqual(dict(all_tested.root_action_values), {-1: 0.0})
        self.assertEqual(all_tested.transitions_evaluated, 0)
        with self.assertRaises(ValueError):
            solution.value_at(np.full_like(root, -1))
        invalid = history.copy()
        invalid[0, 1] = 0
        with self.assertRaises(ValueError):
            solution.action_at(invalid)

    def test_stop_tie_and_deterministic_support(self):
        case = canonical_case()
        zero = dict.fromkeys(case.genes, 0.0)
        tied = replace(case, a_gene=zero, b_gene=zero, omega_gene=zero,
                       fixed_cost=0.0, variable_cost=0.0, allele_freqs=zero)
        engine = NuclearInference(tied)
        solution = solve_exact(engine)
        self.assertEqual(solution.root_value, 0.0)
        self.assertEqual(solution.root_action, -1)
        self.assertTrue(all(value == 0 for value in solution.root_action_values.values()))
        self.assertEqual(solution.states_evaluated, 7)
        self.assertEqual(solution.transitions_evaluated, 9)
        self.assertEqual(solution.terminal_actions_closed, 3)
        impossible = engine.observe(engine.root_state, 0, np.array([1, 0]))
        with self.assertRaises(ValueError):
            solve_exact(engine, impossible)
        with self.assertRaises(ValueError):
            solution.value_at(impossible)

    def test_limits_raise_without_returning_a_partial_solution(self):
        for field, bound, counts in (("max_states", 1, (1, 1)),
                                     ("max_transitions", 1, (2, 1)),
                                     ("max_seconds", 1e-300, (0, 0))):
            with self.subTest(limit=field):
                with self.assertRaises(ExactLimitExceeded) as caught:
                    solve_exact(NuclearInference(canonical_case()), limits=ExactLimits(**{field: bound}))
                error = caught.exception
                self.assertEqual(error.reason, field)
                self.assertEqual((error.states_evaluated, error.transitions_evaluated), counts)
                self.assertGreaterEqual(error.elapsed_seconds, 0)

    def test_invalid_limits(self):
        for field in ("max_states", "max_transitions"):
            for value in (0, -1, 1.5, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ExactLimits(**{field: value})
        for value in (0.0, -1.0, np.inf, np.nan, True):
            with self.subTest(seconds=value), self.assertRaises(ValueError):
                ExactLimits(max_seconds=value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
