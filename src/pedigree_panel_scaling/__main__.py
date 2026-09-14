"""Solve a pedigree exactly, or explicitly simulate a greedy baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .model import load_case, synthetic_case
from .nuclear import make_inference
from .policies import choose_action
from .exact import ExactLimits, ExactLimitExceeded, solve_exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="JSON case; replaces generated family/people/genes")
    parser.add_argument("--people", type=int, default=3)
    parser.add_argument("--genes", type=int, default=2)
    parser.add_argument("--family", choices=("nuclear", "multigeneration"), default="nuclear")
    parser.add_argument("--policy", choices=("exact", "recursive_myopic", "information_greedy"),
                        default="exact")
    parser.add_argument("--max-states", type=int, default=None,
                        help="maximum entered states (default: unlimited)")
    parser.add_argument("--max-transitions", type=int, default=None,
                        help="maximum panel transitions (default: unlimited)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="maximum solver seconds (default: unlimited)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress", action="store_true",
                        help="report exact-solver progress as JSON on stderr")
    args = parser.parse_args()
    try:
        case = load_case(args.case) if args.case else synthetic_case(args.people, args.genes, args.family)
        limits = ExactLimits(args.max_states, args.max_transitions, args.max_seconds)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    started = time.perf_counter()
    engine = make_inference(case)
    observations = engine.root_state
    root = engine.state(observations)
    if args.policy == "exact":
        def report_progress(snapshot):
            print(json.dumps({"status": "running", "states_evaluated": snapshot.states_evaluated,
                "transitions_evaluated": snapshot.transitions_evaluated,
                "independent_states_closed": snapshot.independent_states_closed,
                "solver_seconds": snapshot.elapsed_seconds}), file=sys.stderr, flush=True)
        try:
            solution = solve_exact(engine, limits=limits,
                                   progress=report_progress if args.progress else None)
        except ExactLimitExceeded as error:
            print(json.dumps({
                "status": "resource_limit", "optimal_value": None, "optimal_action": None,
                "reason": error.reason, "people": engine.n_people, "genes": engine.n_genes,
                "states_evaluated": error.states_evaluated,
                "transitions_evaluated": error.transitions_evaluated,
                "solver_seconds": error.elapsed_seconds,
            }, indent=2, allow_nan=False))
            raise SystemExit(2) from error
        def name(action: int) -> str:
            return "STOP" if action < 0 else engine.people[action]
        print(json.dumps({
            "status": "optimal", "policy": "exact", "people": engine.n_people,
            "genes": engine.n_genes, "engine": type(engine).__name__,
            "treewidth": engine.graphical_width,
            "optimal_value": solution.root_value, "optimal_action": name(solution.root_action),
            "root_action_values": {name(a): v for a, v in solution.root_action_values.items()},
            "states_evaluated": solution.states_evaluated,
            "transitions_evaluated": solution.transitions_evaluated,
            "terminal_actions_closed": solution.terminal_actions_closed,
            "independent_states_closed": solution.independent_states_closed,
            "solver_seconds": solution.elapsed_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        }, indent=2, allow_nan=False))
        return
    # Hidden genotypes supply outcomes only AFTER a policy chooses a person.
    world = engine.sample_worlds(observations, 1, np.random.default_rng(args.seed))[0]
    steps = []
    for _ in range(engine.n_people + 1):
        action = choose_action(engine, observations, args.policy)
        if action == -1:
            break
        state = engine.state(observations)
        panel = world[:, action]
        cost = case.fixed_cost + case.variable_cost * bool(np.any(panel > 0))
        steps.append({"person": engine.people[action], "observed_panel": panel.tolist(),
                      "expected_panel_cost_before_test": float(state.test_costs[action]),
                      "realized_panel_cost": float(cost)})
        observations = engine.observe(observations, action, panel)
    else:
        raise RuntimeError("Policy did not terminate within the test-once limit")
    final = engine.state(observations)
    print(json.dumps({
        "purpose": "single synthetic installation example",
        "people": engine.n_people, "genes": engine.n_genes,
        "engine": type(engine).__name__, "treewidth": engine.graphical_width,
        "largest_clique_people": max(len(c.members) for c in engine.tree.cliques),
        "policy": args.policy, "seed": args.seed,
        "initial_residual_risk": root.potential,
        "new_tests": len(steps), "steps": steps,
        "terminal_action": "STOP", "terminal_residual_risk": final.potential,
        "realized_test_cost": sum(step["realized_panel_cost"] for step in steps),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
