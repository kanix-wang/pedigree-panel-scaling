# Pedigree Panel Scaling

A standalone Python package for sequential genetic panel testing, with exact
pedigree inference and full-horizon Bellman optimization. It extracts the
inference core from `gt_code4` and implements the unrestricted backward-induction
recurrence used by `gt_code2`.

The scaling target is **10–15 individuals with 10–15 genes per panel**. Exact
optimization is implemented, but completion at that target size has not been
demonstrated. The default command solves a small three-person, two-gene example;
larger examples also support explicitly selected greedy policies.

## Install and run

Use Python 3.11 or later. From this repository's directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pedigree_panel_scaling
```

Installation uses NumPy 2.3.4 and setuptools 68 or later. On Windows, activate
the environment with `.venv\Scripts\activate`.

The default is `--policy exact --people 3 --genes 2 --family nuclear`. A completed
solve reports `status: "optimal"`, the optimal expected value, the first action,
and every root action's value. The exact solver includes all legal test actions
and supported panel outcomes. It does not sample outcomes or substitute a greedy
policy when work is unfinished.

Run the supplied small case or set resource limits explicitly:

```bash
python -m pedigree_panel_scaling --case examples/trio_2genes.json
python -m pedigree_panel_scaling --case examples/trio_5genes.json
python -m pedigree_panel_scaling --case examples/siblings_4genes.json --max-states 400000 --max-seconds 90
python -m pedigree_panel_scaling --people 3 --genes 3 --max-states 100000 --max-transitions 2000000 --max-seconds 60
```

The defaults allow 100,000 entered states, 2,000,000 enumerated transitions, and
60 solver seconds. On a limit, the command reports `status: "resource_limit"`,
sets the optimal value and action to `null`, and exits with code 2. Time checks
are cooperative between operations, so the time limit can be exceeded by an
operation already running. Inference-engine construction is outside that budget.

For a single simulated trajectory at the target dimensions, select a greedy
policy explicitly:

```bash
python -m pedigree_panel_scaling --people 15 --genes 15 --family nuclear --policy information_greedy --seed 7
python -m pedigree_panel_scaling --people 12 --genes 12 --family multigeneration --policy information_greedy --seed 7
python -m pedigree_panel_scaling --case examples/nuclear_15x15.json --policy recursive_myopic --seed 7
```

These commands stop when the chosen heuristic stops or all individuals have
been tested. The seed makes a trajectory repeatable. A single trajectory is not
an estimate of expected policy performance or evidence that exact optimization
completed at that size.

## What is included

| Path | Purpose |
| --- | --- |
| `src/pedigree_panel_scaling/` | Model, exact inference, exact optimizer, two greedy baselines, and CLI |
| `examples/trio_2genes.json` | Small exact-solver example: two parents and one child, two genes |
| `examples/trio_4genes.json`, `examples/trio_5genes.json` | Three-person exact-solver examples with four and five genes |
| `examples/siblings_4genes.json` | Four-person, four-gene exact example; needs a larger state budget |
| `examples/nuclear_10x10.json` | Synthetic two-parent family: 10 individuals, 10 genes |
| `examples/multigeneration_12x12.json` | Synthetic multigeneration family: 12 individuals, 12 genes |
| `examples/nuclear_15x15.json` | Synthetic two-parent family: 15 individuals, 15 genes |
| `checks/` | Small software correctness checks |
| `docs/model.md` | Model assumptions, equations, and scaling limits |
| `docs/validation.md` | Completed checks and observed execution limits |
| `SOURCE_PROVENANCE.json` | Source extraction record |

Active research experiments, experimental policy variants, benchmark outputs,
research tickets, and historical result reports are outside this repository.
All supplied cases are synthetic examples.

## Use your own family

Copy an example JSON file and replace its people, relationships, genes, and
parameters. A relationship is `[child, parent1, parent2]`; an individual without
a relationship row is a founder. List parents before their children in `people`.
Every parameter map has one entry per gene.
Optional `initial_evidence` contains `[person, panel]` pairs, where each panel is
a list of genotypes in the same order as `genes`.

Genotypes are `0`, `1`, or `2` copies of the modeled allele. A test observes all
genes for one individual, and each individual can be tested once. The package
assumes independence across genes; pedigree relationships create dependence
between relatives within a gene.

The synthetic generator accepts nuclear families with at least three individuals,
multigeneration families with at least five individuals, and at least one gene.
Its illustrative allele-frequency profile repeats every 15 genes. These input
ranges do not guarantee that exact optimization will finish.

## Python interface

Given a `PedigreeCase` named `case`, solve the full remaining decision problem:

```python
from pedigree_panel_scaling import ExactLimits, make_inference, solve_exact

engine = make_inference(case)
solution = solve_exact(
    engine,
    limits=ExactLimits(max_states=100000, max_transitions=2000000, max_seconds=60),
)
value = solution.root_value
action = solution.root_action
action_values = solution.root_action_values
# Actions are indices into case.people; -1 means STOP.
```

`make_inference` selects the nuclear-family specialization when applicable and
uses the general `PedigreeInference` engine otherwise. `solve_exact` accepts an
optional observation array as its second argument and raises
`ExactLimitExceeded` if a resource limit prevents completion. A completed
`ExactSolution` provides `value_at(observations)` and `action_at(observations)`
for supported states in the solved observation graph.

For a greedy action instead:

```python
from pedigree_panel_scaling import choose_action

observations = engine.root_state
action = choose_action(engine, observations, policy="information_greedy")
```

After a selected person's panel is available, update observations with
`engine.observe(observations, action, panel)`. Query the completed exact solution
for the next optimal action, or call `choose_action` again for the chosen greedy
baseline.

`information_greedy` uses the expected one-test reduction in family uncertainty,
weighted by the model's risk coefficients, minus the complete expected panel
cost. `recursive_myopic` compares each immediate test reward with the current
stop reward and repeats that rule after each observation. See
[the model documentation](docs/model.md) for their precise meanings.

## Check the installation

```bash
python -m unittest discover -s checks
```

The supplied target examples contain 10–15 individuals and 10–15 genes.
Current-state inference depends on pedigree graph width; exact optimization also
has to explore joint panel outcomes and future observation states. Only a
completed exact solve establishes the optimum for that case. See
[the validation record](docs/validation.md) for checks and measured limits.
