# Package verification

Verified on 2026-09-14 with Python 3.13.5 and NumPy 2.3.4. The exact optimizer
agrees with native `gt_code2` backward induction on the small cases checked.
Exact completion for 10–15 individuals and 10–15 genes has not been demonstrated.

## Standalone software checks

The suite has ten passing checks: six for exact optimization and four for
inference and greedy trajectory execution. They cover:

- Exact backward induction against an independent complete-world enumeration
  for a two-gene trio, including root action values and continuation queries.
- The two-gene sibling case, optimal parent tie, conditional initial evidence,
  analytical last-person closure, and fully tested terminal states.
- Deterministic support, STOP ties, invalid limits, and each resource-limit
  exception returning no partial solution.
- Independent per-gene posterior and pair-probability enumeration for a
  five-person case, the one-test expected-risk identity, complete-panel rules,
  preservation of initial evidence, impossible evidence, and panel costs.
- Eighteen target-dimension combinations: 10, 12, and 15 individuals crossed
  with 10, 12, and 15 genes, for nuclear and multigeneration pedigrees. These
  check inference and the two greedy policies, including legal termination and
  agreement between general and specialized nuclear calculations at the
  checked histories. They do not run full exact optimization at those sizes.

To repeat the software checks after installation:

```bash
python -m unittest discover -s checks -v
```

The version 0.2.0 wheel was built and installed into a separate directory. From
outside the source checkout, isolated Python passed all ten checks and eleven
command-line checks: the default exact solve, the supplied two-gene trio,
all three resource-limit responses, and all three target-size examples with
each greedy policy. Every installed Python source file matched the checked
source byte-for-byte. Neither `genetic_dp` nor `bayes_risk_audit` was imported.

## Native backward-induction comparison

An external audit compared the packaged solver with `gt_code2`'s
`solve_exact_dp_primal` across every supported full-panel observation state in
two cases. An independent complete-world support census confirmed coverage.

| Case | Supported observation states | Packaged optimal root value | Selected root action |
| --- | ---: | ---: | --- |
| Two parents and one child, two genes | 432 | -0.059806726254195516 | Child |
| Two parents and two children, two genes | 2,264 | -0.0800758734066076 | Father; Mother also optimal |

Across all **2,696 supported states**, the maximum absolute value difference
from native backward induction was **5.55 × 10⁻¹⁷**. Every packaged action was
optimal under the native reference: maximum selected-action regret was zero,
and the independent full-horizon comparisons found zero optimal-action-set
mismatches. The support totals include terminal histories; they are distinct
from the solver's entered-state counters, which avoid enumerating last-person
terminal outcomes.

## Observed exact-solver sizes and limits

The following are completed solves of synthetic nuclear families using the
audit's repeated two-gene parameter profile. They use the full Bellman
recurrence, not simulated trajectories. Times are single-run solver times in
the verification environment, excluding inference-engine construction.

| Individuals | Genes | Entered states | Enumerated transitions | Solver seconds | Optimal root value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 207 | 385 | 0.027 | -0.059806726254195516 |
| 3 | 3 | 1,497 | 2,911 | 0.197 | -0.07868168683247073 |
| 3 | 4 | 11,607 | 22,969 | 1.529 | -0.09702500946239907 |
| 3 | 5 | 93,393 | 186,055 | 12.173 | -0.11363928296019181 |
| 4 | 2 | 1,423 | 3,836 | 0.203 | -0.0800758734066076 |
| 4 | 3 | 19,515 | 55,496 | 2.859 | -0.10652510336941112 |
| 4 | 4 | 291,343 | 850,652 | 41.747 | -0.1335581399291785 |

The first six rows used limits of 100,000 states, 2,000,000 transitions, and
60 seconds. The four-individual, four-gene case initially reached a 10-second
limit; a follow-up with 400,000 states, 2,000,000 transitions, and 90 seconds
completed as shown above.

The exact input files for the three-person four- and five-gene cases and the
four-person four-gene case are included as `examples/trio_4genes.json`,
`examples/trio_5genes.json`, and `examples/siblings_4genes.json`. The README
shows the larger resource budget needed for the sibling example.

Larger cases returned `resource_limit` with no optimal value or action:

| Individuals | Genes | Active limit | Entered states | Enumerated transitions | Solver seconds |
| ---: | ---: | --- | ---: | ---: | ---: |
| 10 | 4 | 100,000 states | 100,000 | 498,632 | 19.551 |
| 10 | 5 | 100,000 states | 100,000 | 498,632 | 20.130 |
| 10 | 10 | 3 seconds | 11,746 | 58,183 | 3.000 |
| 15 | 15 | 3 seconds | 6,905 | 42,903 | 3.000 |

The 10-individual, four- and five-gene follow-ups allowed 30 seconds and
2,000,000 transitions, but reached the 100,000-state cap first. Their earlier
five-second probes had reached the time limit. The two target-size probes
allowed 100,000 states and 2,000,000 transitions but only three seconds; they
establish noncompletion within those short budgets, not infeasibility with
larger budgets. Time limits are cooperative and can be slightly exceeded.

In particular, completion for **three individuals and four or five genes**
does not imply completion for **ten individuals and four or five genes**.
These measurements establish no universal gene ceiling. Family size, supported
observations, pedigree structure, and the resource budget all matter. No exact
optimum is claimed for the 10–15-individual, 10–15-gene scaling target.

## Verification provenance

The 2026-09-14 comparison and size measurements were recorded externally in
`result.json`, `scaling_limits.json`, and `scaling_limits_extended.json` under
the maintainer's `verification/pedigree_panel_scaling_exact_20260914` directory.
All three records are marked complete and retain source-file hashes; their
source-preservation checks passed. Raw verification outputs and research
experiments are not distributed in this package.

The external `installed_check.json` and installation log record the fresh
version 0.2.0 installation checks described above.

The following checks belong to the **historical 2026-09-13 extraction**, before
the exact optimizer was added:

- Syntax-tree comparisons, ignoring documentation and the documented import
  and public-name changes, matched the inference and nuclear-family numerical
  implementations to their source. The junction-tree comparison also accounted
  for removal of unused separator-partition definitions and exports.
- The original four standalone checks passed, including the eighteen
  inference/greedy dimension combinations.
- A wheel installed into a separate directory passed those four checks and ran
  the three then-supplied target-size JSON examples with both greedy policies
  from outside the source repository. It imported neither `bayes_risk_audit`
  nor `genetic_dp`. That historical result does not verify the later exact
  optimizer's wheel installation.

These checks establish the reported software behavior for the recorded cases.
They do not measure comparative speed or validate arbitrary pedigree topologies.
Other supported Python versions were not tested in this verification pass.
