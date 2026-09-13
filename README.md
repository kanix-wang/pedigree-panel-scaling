# Pedigree Panel Scaling

A small, standalone Python package for sequential genetic panel testing in
families of **10–15 individuals with 10–15 genes per panel**. It extracts the
pedigree inference and simple decision policies from `gt_code4` into a repository
that can be shared and run independently.

The package computes exact posterior quantities at the current observed state,
then chooses whether to test another person's complete panel. It includes a
general pedigree engine and a specialized engine for two parents and their
children. The decision policies are heuristics; this package does not solve the
full optimal sequential decision problem.

## Install and run

Use Python 3.11 or later. From this repository's directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pedigree_panel_scaling --people 15 --genes 15 --family nuclear --policy information_greedy --seed 7
```

Installation uses NumPy 2.3.4 and setuptools 68 or later. On Windows, activate
the environment with `.venv\Scripts\activate`.

The command generates a synthetic family and runs one simulated trajectory,
ending when the policy stops or all individuals have been tested. The seed makes
the example repeatable. Its output describes that trajectory; it is not an
estimate of expected policy performance.

Run a multigeneration family or a supplied case:

```bash
python -m pedigree_panel_scaling --people 12 --genes 12 --family multigeneration --seed 7
python -m pedigree_panel_scaling --case examples/nuclear_10x10.json --seed 7
python -m pedigree_panel_scaling --case examples/nuclear_15x15.json --policy recursive_myopic --seed 7
```

## What is included

| Path | Purpose |
| --- | --- |
| `src/pedigree_panel_scaling/` | Model, exact current-state inference, two policies, and command-line example |
| `examples/nuclear_10x10.json` | Synthetic two-parent family: 10 individuals, 10 genes |
| `examples/multigeneration_12x12.json` | Synthetic multigeneration family: 12 individuals, 12 genes |
| `examples/nuclear_15x15.json` | Synthetic two-parent family: 15 individuals, 15 genes |
| `checks/` | Small software correctness checks |
| `docs/model.md` | Model assumptions, equations, and scaling limits |
| `docs/validation.md` | Checks completed for this extraction |
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

The Python interface exposes `PedigreeCase`, `PedigreeInference`,
`NuclearInference`, `make_inference`, and `choose_action`. Given a
`PedigreeCase` named `case`:

```python
from pedigree_panel_scaling import choose_action, make_inference

engine = make_inference(case)
observations = engine.root_state
state = engine.state(observations)
action = choose_action(engine, observations, policy="information_greedy")
# action is an index into case.people, or -1 to stop.
```

`make_inference` selects the nuclear-family specialization when applicable and
uses the general pedigree engine otherwise. After a selected person's panel is
available, update the observations with `engine.observe(observations, action,
panel)` and choose again.

`information_greedy` uses the expected one-test reduction in family uncertainty,
weighted by the model's risk coefficients, minus the complete expected panel
cost. `recursive_myopic` compares each immediate test reward with the current
stop reward and repeats that rule after each observation. See
[the model documentation](docs/model.md) for their precise meanings.

## Check the installation

```bash
python -m unittest discover -s checks
```

The target examples cover 10–15 individuals and 10–15 genes. General pedigree
cost also depends on the width of its relationship graph: a densely connected
family can be harder than a larger, simpler family. Exact posterior inference
does not imply an optimal testing policy or a bound on the policy's distance
from optimality.
