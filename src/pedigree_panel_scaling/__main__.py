"""Run one reproducible synthetic trajectory as an installation example."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .model import load_case, synthetic_case
from .nuclear import make_inference
from .policies import choose_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="JSON case; replaces generated family/people/genes")
    parser.add_argument("--people", type=int, choices=range(10, 16), default=15)
    parser.add_argument("--genes", type=int, choices=range(10, 16), default=15)
    parser.add_argument("--family", choices=("nuclear", "multigeneration"), default="nuclear")
    parser.add_argument("--policy", choices=("recursive_myopic", "information_greedy"),
                        default="information_greedy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    case = load_case(args.case) if args.case else synthetic_case(args.people, args.genes, args.family)
    if not (10 <= len(case.people) <= 15 and 10 <= len(case.genes) <= 15):
        parser.error("The sharing example is scoped to 10–15 people and 10–15 genes")
    started = time.perf_counter()
    engine = make_inference(case)
    observations = engine.root_state
    root = engine.state(observations)
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
