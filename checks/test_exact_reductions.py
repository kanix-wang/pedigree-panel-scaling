"""Independent oracle and symmetry checks for exact reductions."""

from dataclasses import replace
import unittest

import numpy as np

from pedigree_panel_scaling import NuclearInference, PedigreeCase, PedigreeInference, synthetic_case
from pedigree_panel_scaling.exact_reductions import independent_scores
from pedigree_panel_scaling.exact import _ObservationCodec
from pedigree_panel_scaling._nuclear_exact import _FastModel
from test_exact import ExplicitWorldOracle, canonical_case


class ExactReductionChecks(unittest.TestCase):
    def test_independent_tail_matches_complete_world_backward_induction(self):
        base = canonical_case(siblings=True)
        for panels in ((0, 0), (0, 1), (1, 1)):
            evidence = (("F", (panels[0],) * 2), ("M", (panels[1],) * 2))
            case = replace(base, initial_evidence=evidence)
            oracle = ExplicitWorldOracle(case)
            for kind in (PedigreeInference, NuclearInference):
                engine = kind(case)
                state = engine.state(engine.root_state)
                scores = independent_scores(engine, state)
                expected_value, expected_scores = oracle.solve(oracle.history_key(engine.root_state))
                self.assertEqual(set(scores), set(expected_scores))
                for action in scores:
                    self.assertAlmostEqual(scores[action], expected_scores[action], places=12)
                self.assertAlmostEqual(max(scores.values()), expected_value, places=12)

    def test_independent_mixed_gains_preserve_zero_gain_first_action(self):
        parents = ("Z1", "Z2", "P1", "P2", "N1", "N2")
        children = ("Zero", "Positive", "Negative")
        case = PedigreeCase(
            people=parents + children,
            relationships=tuple((child, *parents[2*i:2*i+2]) for i, child in enumerate(children)),
            genes=("g",), allele_freqs={"g": 0.2}, a_gene={"g": 0.0},
            b_gene={"g": -0.5}, delta_gene={"g": 0.5}, omega_gene={"g": 0.0},
            fixed_cost=3/16, variable_cost=0.0,
            initial_evidence=tuple((parent, (value,)) for parent, value in
                                   zip(parents, (1, 1, 0, 1, 0, 0))),
        )
        engine = PedigreeInference(case)
        scores = independent_scores(engine, engine.state(engine.root_state))
        expected = {-1: -7/16, 6: -3/8, 7: -3/8, 8: -9/16}
        oracle = ExplicitWorldOracle(case)
        _, oracle_scores = oracle.solve(oracle.history_key(engine.root_state))
        for action in expected:
            self.assertAlmostEqual(scores[action], expected[action], places=14)
            self.assertAlmostEqual(scores[action], oracle_scores[action], places=12)
        self.assertEqual(max(scores, key=scores.__getitem__), 6)

    def test_guard_requires_graph_separation_and_handles_terminal_state(self):
        engine = NuclearInference(canonical_case())
        root = engine.root_state
        self.assertIsNone(independent_scores(engine, engine.state(root)))
        one_parent = engine.observe(root, 0, np.array([0, 1]))
        self.assertIsNone(independent_scores(engine, engine.state(one_parent)))
        both = engine.observe(one_parent, 1, np.array([1, 1]))
        self.assertIsNotNone(independent_scores(engine, engine.state(both)))
        complete = engine.observe(both, 2, np.array([0, 1]))
        self.assertEqual(independent_scores(engine, engine.state(complete)), {-1: 0.0})
        tied = replace(canonical_case(), allele_freqs={"a": 0.0, "b": 0.0})
        deterministic = NuclearInference(tied)
        scores = independent_scores(deterministic, deterministic.state(deterministic.root_state))
        self.assertEqual(scores[-1], 0.0)
        self.assertTrue(all(scores[person] == -tied.fixed_cost for person in range(3)))

    def test_exact_support_guard_rejects_rounded_point_masses_and_gene_mixing(self):
        tiny = replace(canonical_case(), allele_freqs={"a": 1e-17, "b": 1e-17})
        engine = NuclearInference(tiny)
        state = engine.state(engine.root_state)
        self.assertEqual(float(state.genotype.max()), 1.0)
        self.assertTrue(np.all(np.count_nonzero(state.genotype > 0, axis=-1) > 1))
        self.assertIsNone(independent_scores(engine, state))
        model = _FastModel(engine)
        self.assertFalse(model.summary(model.key(engine.root_state))[3])
        mixed = NuclearInference(replace(tiny, allele_freqs={"a": 0.0, "b": 0.2}))
        state = mixed.state(mixed.root_state)
        self.assertTrue(np.all(np.count_nonzero(state.genotype[0] > 0, axis=-1) == 1))
        self.assertIsNone(independent_scores(mixed, state))
        model = _FastModel(mixed)
        self.assertFalse(model.summary(model.key(mixed.root_state))[3])

    def test_distinct_counts_merge_equal_posteriors_and_close_inferred_parents(self):
        case = synthetic_case(6, 1)
        oracle = ExplicitWorldOracle(case)
        for kind in (PedigreeInference, NuclearInference):
            engine = kind(case)
            codec = _FastModel(engine) if kind is NuclearInference else _ObservationCodec(engine)
            histories = []
            for outcomes in ((0, 1, 2), (0, 0, 2)):
                history = engine.root_state
                for child, value in zip((2, 3, 4), outcomes):
                    history = engine.observe(history, child, np.array([value]))
                histories.append(history)
            if kind is NuclearInference:
                self.assertEqual(codec.key(histories[0]), codec.key(histories[1]))
            else:
                self.assertNotEqual(codec.key(histories[0]), codec.key(histories[1]))
            for history in histories:
                state = engine.state(history)
                self.assertFalse(engine.independent_remaining(history))
                np.testing.assert_array_equal(state.genotype[0, :2], [[0, 1, 0], [0, 1, 0]])
                scores = independent_scores(engine, state)
                expected_value, expected = oracle.solve(oracle.history_key(history))
                for action in expected:
                    self.assertAlmostEqual(scores[action], expected[action], places=12)
                self.assertAlmostEqual(max(scores.values()), expected_value, places=12)
                if kind is NuclearInference:
                    _, _, compact_scores, closed = codec.summary(codec.key(history))
                    self.assertTrue(closed)
                    for action, value in codec.public_scores(compact_scores, history).items():
                        self.assertAlmostEqual(value, expected[action], places=12)
            if kind is NuclearInference:
                parent_tested = engine.observe(histories[0], 0, np.array([1]))
                self.assertNotEqual(codec.key(histories[0]), codec.key(parent_tested))
                np.testing.assert_array_equal(engine.state(histories[0]).genotype,
                                              engine.state(parent_tested).genotype)

    def test_known_parent_and_mixed_children_have_exact_posterior_signature(self):
        case = synthetic_case(6, 1)
        engine = NuclearInference(case)
        codec = _FastModel(engine)
        histories = []
        for outcomes in ((0, 0, 1), (0, 1, 1)):
            history = engine.observe(engine.root_state, 0, np.array([0]))
            for child, value in zip((2, 3, 4), outcomes):
                history = engine.observe(history, child, np.array([value]))
            histories.append(history)
        self.assertEqual(codec.key(histories[0]), codec.key(histories[1]))
        first, second = (engine.state(history) for history in histories)
        np.testing.assert_array_equal(first.genotype[:, ~first.tested], second.genotype[:, ~second.tested])
        self.assertEqual(independent_scores(engine, first), independent_scores(engine, second))
        wrong_parent = engine.observe(histories[0], 1, np.array([0]))
        with self.assertRaisesRegex(ValueError, "zero support"):
            codec.key(wrong_parent)
        fixed = NuclearInference(replace(case, allele_freqs={"g1": 0.0}))
        impossible_child = fixed.observe(fixed.root_state, 2, np.array([1]))
        with self.assertRaisesRegex(ValueError, "zero support"):
            _FastModel(fixed).key(impossible_child)

    def test_child_counts_ignore_past_panel_pairing_and_remap_child_actions(self):
        base = canonical_case(siblings=True)
        case = replace(base, people=(*base.people, "E"),
                       relationships=(*base.relationships, ("E", "F", "M")))
        engine = NuclearInference(case)
        codec = _FastModel(engine)
        first = engine.observe(engine.root_state, 2, np.array([0, 0]))
        first = engine.observe(first, 3, np.array([1, 1]))
        rearranged = engine.observe(engine.root_state, 2, np.array([0, 1]))
        rearranged = engine.observe(rearranged, 3, np.array([1, 0]))
        renamed = engine.observe(engine.root_state, 3, np.array([0, 0]))
        renamed = engine.observe(renamed, 4, np.array([1, 1]))
        self.assertEqual(codec.key(first), codec.key(rearranged))
        self.assertEqual(codec.key(first), codec.key(renamed))
        states = [engine.state(history) for history in (first, rearranged, renamed)]
        for other in states[1:]:
            np.testing.assert_allclose(states[0].genotype[:, ~states[0].tested],
                                       other.genotype[:, ~other.tested], atol=1e-14)
            np.testing.assert_allclose(states[0].test_costs[~states[0].tested],
                                       other.test_costs[~other.tested], atol=1e-14)
            np.testing.assert_allclose(states[0].test_rewards[~states[0].tested],
                                       other.test_rewards[~other.tested], atol=1e-14)
            self.assertAlmostEqual(states[0].stop_reward, other.stop_reward, places=14)
        self.assertEqual(codec.decode_action(2, first), 4)
        self.assertEqual(codec.decode_action(2, renamed), 2)
        self.assertEqual(codec.decode_action(0, first), 0)
        self.assertEqual(codec.decode_action(-1, first), -1)
        changed_parent = engine.observe(first, 0, np.array([1, 1]))
        self.assertNotEqual(codec.key(first), codec.key(changed_parent))
        with self.assertRaises(ValueError):
            codec.decode_action(0, changed_parent)
        all_children = engine.observe(first, 4, np.array([0, 0]))
        with self.assertRaises(ValueError):
            codec.decode_action(2, all_children)

    def test_gene_permutations_require_identical_complete_parameter_profiles(self):
        base = canonical_case(siblings=True)
        genes = ("x", "y", "z")
        case = replace(base, genes=genes, allele_freqs={"x": 0.1, "y": 0.1, "z": 0.2},
                       a_gene=dict.fromkeys(genes, -0.08), b_gene=dict.fromkeys(genes, -0.04),
                       delta_gene=dict.fromkeys(genes, 0.6), omega_gene=dict.fromkeys(genes, 0.0))
        engine = NuclearInference(case)
        codec = _FastModel(engine)
        observed = engine.observe(engine.root_state, 2, np.array([0, 1, 2]))
        swapped = observed[[1, 0, 2]]
        self.assertEqual(codec.key(observed), codec.key(swapped))
        self.assertNotEqual(codec.key(observed), codec.key(observed[[2, 1, 0]]))
        first, second = engine.state(observed), engine.state(swapped)
        np.testing.assert_allclose(second.genotype, first.genotype[[1, 0, 2]], atol=1e-14)
        np.testing.assert_allclose(first.test_costs, second.test_costs, atol=1e-14)
        np.testing.assert_allclose(first.test_rewards, second.test_rewards, atol=1e-14)
        self.assertAlmostEqual(first.stop_reward, second.stop_reward, places=14)
        for parameter in ("a_gene", "b_gene", "delta_gene", "omega_gene"):
            mapping = dict(getattr(case, parameter))
            mapping["y"] += 0.01
            different = _FastModel(NuclearInference(replace(case, **{parameter: mapping})))
            self.assertNotEqual(different.key(observed), different.key(swapped))
        mutable_input = replace(case, allele_freqs=dict(case.allele_freqs))
        captured = NuclearInference(mutable_input)
        mutable_input.allele_freqs["z"] = 0.1
        captured_codec = _FastModel(captured)
        self.assertNotEqual(captured_codec.key(observed), captured_codec.key(observed[[2, 1, 0]]))

    def test_generic_keys_and_actions_keep_person_identity_and_validate_evidence(self):
        engine = PedigreeInference(canonical_case(siblings=True))
        codec = _ObservationCodec(engine)
        first = engine.observe(engine.root_state, 2, np.array([0, 1]))
        other = engine.observe(engine.root_state, 3, np.array([0, 1]))
        self.assertEqual(codec.key(first), first.tobytes())
        self.assertNotEqual(codec.key(first), codec.key(other))
        self.assertEqual(codec.decode_action(3, first), 3)
        with self.assertRaises(ValueError):
            codec.decode_action(-2, first)
        partial = first.copy()
        partial[0, 0] = 0
        with self.assertRaises(ValueError):
            codec.key(partial)
        initial = NuclearInference(replace(canonical_case(), initial_evidence=(("C", (0, 1)),)))
        with self.assertRaises(ValueError):
            _FastModel(initial).key(np.full_like(initial.root_state, -1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
