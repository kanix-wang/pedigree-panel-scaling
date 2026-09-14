# Package verification

The exact optimizer uses the same unrestricted Bellman problem as native
`gt_code2` backward induction. Default state, transition, and time limits have
been removed. The completed runs below use no resource caps. Completion for
10–15 individuals with 10–15 genes has not been demonstrated.

## Numerical compatibility

The external 2026-09-14 audit compares optimal values and chosen-action values
throughout both complete supported two-gene observation graphs:

| Family | Supported histories | Native optimal root value | Root action |
| --- | ---: | ---: | --- |
| Two parents and one child | 432 | -0.05980672625419552 | Child |
| Two parents and two children | 2,264 | -0.08007587340660761 | Either parent |

The installed version 0.3.0 native solver agrees on all **2,696 supported
histories**, with maximum value discrepancy **2.78 × 10⁻¹⁷** and zero
selected-action regret. The earlier observation-based reduced implementation
had maximum discrepancy 5.55 × 10⁻¹⁷ on the same histories. The reference is `genetic_dp.exact_dp.solver.solve_exact_dp_primal`.
A separate full-joint Bellman oracle and an unrestricted Gurobi linear program
also cross-check the roots. The multigene dual using additive per-gene value
functions is an upper-bound calculation and is not the exact reference here.

The per-gene-state implementation was additionally compared with independent
complete-world enumeration on two four-person, two-gene cases: identical gene
profiles and a deterministic/nondegenerate pair. For this implementation alone,
all **2,416 supported histories** pass and **17,584 impossible histories** are
rejected. Every supported-state value and selected-action value, and all root
action values, agree within **7.22 × 10⁻¹⁶**. A separate general-engine comparison
checks the same histories. Counts that sum both engines should not be mistaken
for additional distinct cases.

These are floating-point agreement checks, not a promise of bitwise-identical
values. Mathematically tied actions may be affected by floating-point summation
order; selected-action values are checked against the reference optimum.

## Completed uncapped exact solves

The following measurements use a ten-person nuclear family: two founders and
eight exchangeable children. Both Python and optional C++ implementations retain
the full decision horizon and every supported outcome for expanded actions.
Times are individual solver runs, excluding construction of the inference
engine. Peak memory is the whole process resident high-water mark on macOS,
not just the value table. The machine has 512 GiB installed memory.

| Backend | Genes | Numerical gene profiles | Canonical states | Panel transitions | Solver seconds | Peak MiB | Optimal root value |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Python | 4 | Two profiles, each repeated twice | 613,395 | 28,048,248 | 44.69 | 199.44 | -0.35726014850437826 |
| Python | 5 | Two profiles, repeated three and two times | 4,492,149 | 488,820,671 | 770.82 | 1194.64 | -0.4284641123009654 |
| Python | 4 | Different frequency for every gene | 2,213,982 | 100,270,178 | 181.86 | 614.47 | 0.25821118553488315 |
| C++ | 5 | Two profiles, repeated three and two times | 4,492,149 | 488,820,671 | 18.39 | 355.19 | -0.4284641123009654 |
| C++ | 5 | Different frequency for every gene | 44,454,192 | 4,739,339,544 | 231.01 | 2919.11 | 0.31812411826030124 |

The alternating-profile cases use allele frequencies 0.05/0.08, input `a`
coefficients −0.08/−0.06, `b` coefficients −0.04/−0.03, `delta` 0.6/0.7,
zero `omega`, fixed cost 0.01, and variable cost 0.02. The five-gene case chooses
a child; the four-gene case chooses a parent. Its full input is supplied as
`examples/nuclear_10x5_two_profiles.json`.

The distinct-frequency cases are exactly the package's `--people 10 --genes 4`
and `--people 10 --genes 5` synthetic examples. Both select Father, with Mother
equally optimal. Its other parameters differ from the alternating-profile
case, so these two rows are not a controlled timing comparison of profile
multiplicity alone. Gene-profile equality must include frequency and every
effective reward coefficient; similar parameters do not permit merging.

A matched four-person, five-gene case provides an implementation comparison:

| Implementation | States | Transitions | Solver seconds | Optimal value |
| --- | ---: | ---: | ---: | ---: |
| Original full observation histories, uncapped | 4,544,763 | 13,447,016 | 655.80 | -0.15791697977959407 |
| Observation arrays with exact state reductions | 77,695 | 761,773 | 57.96 | -0.15791697977959407 |
| Cached per-gene states | 77,695 | 761,773 | 1.54 | -0.15791697977959412 |

All root action values agree to at most 5.55 × 10⁻¹⁷. Both parents are optimal.
The input is `examples/siblings_5genes.json`. The optional C++ implementation
also completed the ten-person, five-gene alternating-profile case in 18.39
seconds using 355.19 MiB peak process memory. Its value, every root action
value, state count, and transition count match the Python result bit-for-bit
for this case. The reductions alter the number
of represented states, not the underlying probability model or policy choices.
The original run demonstrates actual completion beyond the previous arbitrary
100,000-state cap; that cap supplied no evidence of mathematical infeasibility.

## Remaining scale limitations

There is no hard-coded gene ceiling. Runtime and memory depend on family size,
initial evidence, probability support, topology, and the number of distinct
numerical gene profiles. The exact nuclear solver combines equivalent child
histories, but still retains a joint value table over gene posterior states.

An integer structural census establishes the following state counts for the
current expansion and independence-closure rules. Assumptions are no initial
evidence, positive founder support preserved numerically, exchangeable children,
and maximal groups of exactly equal numerical gene profiles:

| Individuals | Genes | Alternating two profiles | All gene profiles distinct |
| ---: | ---: | ---: | ---: |
| 10 | 5 | 4,492,149 | 44,454,192 |
| 10 | 10 | 34,682,522,428 | 215,509,653,745,254 |
| 15 | 15 | 6,349,458,510,000,527 | 314,219,910,308,597,426,831,637 |

These are structural counts, not completed large solves or runtime estimates.
For ten people and ten distinct gene profiles, even eight bytes per value alone
would require about 1.72 petabytes, before keys, actions, or table overhead.
That complete-table representation cannot fit this machine's memory. Even the
alternating-profile ten-gene case requires an 86-bit packed key and therefore
uses the Python backend. On this 64-bit Python build each twelve-element key
tuple occupies 136 bytes; its 34,682,522,428 keys alone require about 4.72 TB,
before values or dictionary storage. This is
not an impossibility result for every exact algorithm. Removing default limits
does not establish completion at 10–15 genes.

## Software checks and provenance

Run the standalone checks after installation:

```bash
python -m unittest discover -s checks -v
```

A fresh version 0.3.0 wheel with the native extension was installed into a
separate directory and tested outside the source checkout using isolated Python.
All **30 software checks** passed with no skips, as did **16 command-line
checks**. These include the default exact solve, explicit limit responses,
progress output, the supplied small exact cases, the uncapped ten-person
five-gene alternating-profile case, and the three large examples under both
explicit greedy policies. That installed five-gene command completed in 17.73
seconds with 4,492,149 states and 488,820,671 transitions. A separate uncapped
run of the installed native solver completed the ten-person case with five
distinct gene profiles in **233.05 seconds** and 2,916.30 MiB peak process
memory. Its value, all root action values, 44,454,192 states and 4,739,339,544
transitions match the external native result exactly for this case.
All installed Python
source bytes match the checked source; neither `genetic_dp` nor
`bayes_risk_audit` was imported. Python 3.13.5, NumPy 2.3.4, and the macOS arm64
native build were tested. Other supported Python/platform combinations were
not exercised in this pass.

Both the Python and compiled backends passed the independent exhaustive oracle
checks described above at twelve decimal places. The suite also verifies
missing or unloadable native extensions and wider-key fallback, explicit limits,
read-only value/action maps, lazy iteration, and preservation of observation
queries. A separate compiler-failure check confirms installation can proceed
without the optional extension.

The checks cover independent enumeration, complete-panel rules, conditional
initial evidence, root and descendant queries, exact state equivalence,
deterministic support, impossible evidence, independent and last-person
closures, ties, progress callbacks, and explicitly requested resource limits.
Eighteen dimension combinations (10/12/15 people, 10/12/15 genes, nuclear and
multigeneration families) check inference and both greedy policies. Those
large-dimension checks do not solve their full optimization problems.

The 2026-09-14 capacity records are external to this shareable repository, under
`verification/pedigree_panel_scaling_capacity_20260914`. They retain complete
synthetic parameters, final status, counters, time, process memory, source hashes,
and frozen executed source snapshots. Earlier version 0.2.0 results remain in
`verification/pedigree_panel_scaling_exact_20260914`; their imposed budgets are
historical, not current defaults. Raw reports and research experiments are not
distributed with this package.

The reference `gt_code2` checkout has existing local modifications. Its branch,
commit, and actual source-file hashes are recorded, so this is a numerical
compatibility check against that source, not a claim of a clean canonical
benchmark checkout. Neither source repository is a runtime dependency.
