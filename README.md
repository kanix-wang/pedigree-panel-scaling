# Pedigree Panel Scaling

A standalone Python package for sequential genetic panel testing, with exact
pedigree inference and full-horizon Bellman optimization. It extracts the
inference core from `gt_code4` and implements the unrestricted backward-induction
recurrence used by `gt_code2`.

The scaling target is **10–15 individuals with 10–15 genes per panel**. Exact
optimization has completed for the documented ten-person, four- and five-gene
nuclear families. Completion at the full 10–15-gene target has not been
demonstrated. The default command solves a small three-person, two-gene example.

## Install and run

Use Python 3.11 or later. From this repository's directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pedigree_panel_scaling
```

Installation uses NumPy 2.3.4 and setuptools 68 or later. A C++17 compiler enables
the optional native accelerator; if it cannot be built, installation retains
the exact Python solver. On Windows, activate the environment with
`.venv\Scripts\activate`.

The default is `--policy exact --people 3 --genes 2 --family nuclear`. A completed
solve reports `status: "optimal"`, the optimal expected value, the first action,
and every root action's value. The exact solver preserves the full decision
problem while combining equivalent states and actions and closing independent
remaining decisions analytically. It does not sample outcomes or substitute a
greedy policy when work is unfinished.

Run a supplied small case, with optional progress output:

```bash
python -m pedigree_panel_scaling --case examples/trio_2genes.json
python -m pedigree_panel_scaling --case examples/trio_5genes.json --progress
python -m pedigree_panel_scaling --case examples/siblings_5genes.json --progress
python -m pedigree_panel_scaling --people 10 --genes 5 --progress
python -m pedigree_panel_scaling --case examples/nuclear_10x5_two_profiles.json --progress
python -m pedigree_panel_scaling --people 3 --genes 3
```

There are no default state, transition, or time limits. An exact solve continues
until completion, interruption, or an error. `--progress` writes JSON progress
records to standard error at most once per second; the final result is written
to standard output. Progress counts describe canonical decision states actually
entered and outcome transitions actually enumerated, not the number of distinct
observation histories represented by the solution.

Resource controls are optional. For example, to request a 60-second solver
budget explicitly:

```bash
python -m pedigree_panel_scaling --case examples/siblings_4genes.json --max-seconds 60 --progress
```

The optional flags are `--max-states`, `--max-transitions`, and `--max-seconds`.
On a limit, the command reports `status: "resource_limit"`, sets the optimal value
and action to `null`, and exits with code 2. Time checks are cooperative between
operations, so an operation already running can exceed the requested time.
Inference-engine construction is outside that budget.

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
| `examples/siblings_4genes.json`, `examples/siblings_5genes.json` | Four-person exact-solver examples |
| `examples/nuclear_10x5_two_profiles.json` | Completed ten-person, five-gene exact case with two repeating parameter profiles |
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
from pedigree_panel_scaling import make_inference, solve_exact

engine = make_inference(case)
solution = solve_exact(engine)
value = solution.root_value
action = solution.root_action
action_values = solution.root_action_values
# Actions are indices into case.people; -1 means STOP.
```

`make_inference` selects the nuclear-family specialization when applicable and
uses the general `PedigreeInference` engine otherwise. `solve_exact` accepts an
optional observation array as its second argument and raises
`ExactLimitExceeded` if an explicitly supplied resource limit prevents completion.
All fields of `ExactLimits()` default to `None`, meaning unlimited. To set an
optional budget, pass, for example, `limits=ExactLimits(max_seconds=60)` after
importing `ExactLimits`.

For programmatic progress, pass a callback as `progress=callback`. It receives
an `ExactProgress` snapshot at most once per second with `states_evaluated`,
`transitions_evaluated`, `terminal_actions_closed`, `independent_states_closed`,
and `elapsed_seconds`. The solver itself does not print progress; the completed
solution contains the final counters. Callback exceptions propagate to the caller.

A completed `ExactSolution` provides `value_at(observations)` and
`action_at(observations)` for supported descendants of the solved root. Queries
must preserve initial evidence and any observations supplied to `solve_exact`.
They use the stored canonical states or compute analytically closed descendants
exactly, without starting another search.

For nuclear families, exact state compression retains the parental posterior,
which parents have been tested, and the number of remaining children. Cached
per-gene states avoid repeated pedigree inference inside the search. It combines
exchangeable children and genes with identical numerical profiles. General
pedigrees retain complete observation keys. Both engines can close a remaining
set of conditionally independent individuals analytically. These reductions use
no probability rounding, outcome pruning, or planning-horizon restriction; see
[the model documentation](docs/model.md) for the conditions and formulas.

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
