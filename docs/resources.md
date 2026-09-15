# Measured resources and reproducible projections

These projections describe the existing **exact optimizer**, using completed
experiments from September 14–15, 2026. They concern nuclear families with two
labeled parents and exchangeable children, no initial evidence, and positive
probability support preserved numerically. They do not predict resources for
arbitrary pedigrees. **1,000 GB means 1,000,000,000,000 bytes** (about 931.32 GiB).

## Where 1,000 GB is crossed

The table gives the first integer family size or gene count at which modeled
peak memory exceeds 1,000 GB. In these four queries, even the necessary storage
lower bound crosses at that same integer. These are projections, not completed
large solves or observed allocation failures.

| Fixed dimension | Gene profiles | First crossing | Necessary storage, GB | Central peak projection, GB | Rough compute-only duration |
| --- | --- | --- | ---: | ---: | ---: |
| 15 genes | All distinct | **3 people** | 55,138,892 | 84,880,582 (84.9 PB) | About 20 years |
| 15 people | All distinct | **7 genes** | 14,895 | 18,593 | About 265 days |
| 15 genes | Two alternating profiles | **4 people** | 1,277 | 1,859 | About 229 days |
| 15 people | Two alternating profiles | **8 genes** | 4,807 | 8,268 | About 4.1 years |

Three people is the smallest supported nuclear family. Thus **15 distinct genes
already exceed 1,000 GB at the smallest family**, by a very large margin.
For comparison, the immediately smaller supported cases are:

| People | Genes | Gene profiles | Central peak projection, GB | Rough compute-only duration |
| ---: | ---: | --- | ---: | ---: |
| 15 | 6 | All distinct | 614 | 3.46 days |
| 3 | 15 | Two alternating profiles | 36.3 | 1.23 days |
| 15 | 7 | Two alternating profiles | 168 | 1.53 days |

**The durations are unvalidated extrapolations of measured computation rates,
not completion-time promises.** They assume enough memory and exclude paging.
None of the four crossing cases fits a 1,000 GB machine using the measured
uncompressed table representation. Their actual completion times on such a
machine cannot be inferred from these durations. A projection below 1,000 GB
also does not establish that a case has completed or is guaranteed to fit.

The modeled low–high peak scenarios for the crossing cases are respectively
81.4–101.8 PB, 18,588–19,563 GB, 1,779–2,145 GB, and 7,884–10,301 GB.
These are allocator/calibration sensitivity scenarios, **not statistical
confidence intervals or guaranteed bounds**. The Python projections in
particular extend far beyond the two sub-1-GB calibration runs.

An alternating profile repeats two complete numerical parameter profiles,
with group sizes `ceil(G/2)` and `floor(G/2)`. Equality must include the prior
and all effective reward parameters. Similar genes cannot be merged merely
because their parameters are close. The distinct and alternating benchmark
inputs also differ in costs and rewards, so their timing comparison does not
isolate the effect of gene symmetry alone.

## Fixing ten people or ten genes instead

Ten people is already demonstrated for six distinct genes (59.59 GB, 4.78
hours), and seven genes sharing two profiles (13.25 GB, 1.74 hours). Ten genes
have completed for a three-person family with two alternating profiles
(0.201 GB, 3.08 seconds). These are measured exact completions.

The following first crossings use the central peak-memory projection:

| Fixed dimension | Gene profiles | First crossing | Central peak projection, GB | Rough compute-only duration |
| --- | --- | --- | ---: | ---: |
| 10 people | All distinct | **7 genes** | 1,184 | 10.3 days |
| 10 people | Two alternating profiles | **9 genes** | 2,284 | 643 days |
| 10 genes | All distinct | **4 people** | 8,140 | 5.35 days |
| 10 genes | Two alternating profiles | **8 people** | 2,094 | 958 days |

For ten people/seven distinct genes, necessary storage alone is 952 GB; it is
the measured-overhead peak model that exceeds 1,000 GB. The necessary-storage
criterion first crosses at eight genes. For the other three rows, necessary
storage and the peak scenarios cross at the same integer shown above.

Below these crossings, ten people/eight alternating-profile genes project to
393 GB and about 49.5 days; seven people/ten alternating-profile genes project
to 711 GB and about 266 days. Both use Python because the current native key
exceeds 64 bits. Thus remaining below 1,000 GB does not imply a short runtime.
Three people/ten distinct genes project to 252 GB; the script transfers a
rough 12-minute compute estimate from a 20-person/five-gene run, but neither
dimension matches, so that runtime is particularly weakly supported.

Ten people **and** ten genes remain beyond 1,000 GB: approximately 76.3 PB
with distinct profiles, or 11,754 GB with two alternating profiles. These
dimensions have not completed. The same runtime and memory qualifications
apply as above; these are current-representation projections, not universal
limits on exact optimization.

```bash
python benchmarks/project.py --fixed-people 10 --threshold-gb 1000
python benchmarks/project.py --fixed-genes 10 --threshold-gb 1000
python benchmarks/project.py --fixed-people 10 --threshold-gb 1000 --profile alternating
python benchmarks/project.py --fixed-genes 10 --threshold-gb 1000 --profile alternating
```

## Reproduce the projections without starting an optimizer

From the repository directory, with Python 3.11 or later:

```bash
python benchmarks/project.py --fixed-genes 15 --threshold-gb 1000
python benchmarks/project.py --fixed-people 15 --threshold-gb 1000
python benchmarks/project.py --fixed-genes 15 --threshold-gb 1000 --profile alternating
python benchmarks/project.py --fixed-people 15 --threshold-gb 1000 --profile alternating

python benchmarks/project.py --people 15 --genes 6
python benchmarks/project.py --people 3 --genes 15 --profile alternating
python benchmarks/project.py --people 15 --genes 7 --profile alternating
```

These scripts use only the Python standard library and the included data.
They calculate integer work counts and resource projections; they never import
or run the optimizer. JSON output includes the first crossing and predecessor
for each memory criterion, backend, exact work counts, memory scenarios,
runtime reference case, and the reference's measured time and branch count.
The threshold is a capacity query, not a solver stopping limit.

## Experimental basis

[reference.json](../benchmarks/reference.json) contains **18 completed cases**:
12 nuclear cases and six multigeneration cases. Every case includes the complete
synthetic input, optimum, all root action values, four work counters, solver
time, peak resident memory, backend, and a hash of its original result.
Use `python benchmarks/verify.py --list` for all case IDs and measurements.

Representative original measurements, converted to decimal GB:

| People | Genes | Family / profiles | Backend | Solver time | Peak GB |
| ---: | ---: | --- | --- | ---: | ---: |
| 10 | 4 | Nuclear / distinct | Native | 4.26 seconds | 0.210 |
| 15 | 5 | Nuclear / distinct | Native | 1.13 hours | 21.58 |
| 20 | 5 | Nuclear / distinct | Native | 7.03 hours | 96.03 |
| 10 | 6 | Nuclear / distinct | Native | 4.78 hours | 59.59 |
| 10 | 7 | Nuclear / alternating | Native | 1.74 hours | 13.25 |
| 9 | 2 | Multigeneration / distinct | Python | 4.01 hours | 7.83 |

The reference machine was macOS arm64, Python 3.13.5, NumPy 2.3.4, with 32
logical CPUs and 512 GiB installed memory. The solver source was version 0.3.0,
commit `3963f91317422e051de7ac894f20794c2ec9d526`; source hashes are included.
Independent cases ran concurrently, but **each solve used one core**. Do not
divide a single-case duration by the number of available cores. The portable
verifier runs selected cases sequentially in separate processes.

[python_calibration.json](../benchmarks/python_calibration.json) adds two fresh,
uncapped Python-backend reruns, both matching their reference optima, every root
action value, and all four work counters:

| People | Genes | Profiles | States | Panel branches | Solver seconds | Peak GB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 10 | Alternating | 2,084,581 | 62,639,649 | 132.96 | 0.771 |
| 10 | 4 | Distinct | 2,213,982 | 100,270,178 | 147.26 | 0.634 |

The calibration file retains the reference-file hash actually used when it ran.
Subsequent metadata normalization labeled the historical ten-person/five-gene
reference as nuclear/distinct using the measurement index; neither calibration
case nor any model, optimum, or work count changed. The projector checks the
calibration models, dimensions, counts, and numerical results against the
current reference data before using them.

## What is calculated and what is estimated

**State and branch counts.** [census.py](../benchmarks/census.py) enumerates
per-gene posterior likelihood signatures using integer powers of two and exact
support, then combines equal-profile genes as multisets. It applies the existing
terminal and conditional-independence closure rule. Its state, panel-branch,
terminal-closure, and independent-closure counts match all 12 measured nuclear
cases. This is an integer census under the assumptions at the top of this page,
not a new solve at the projected dimensions. Degenerate priors, initial evidence,
other topologies, lost numerical support, or changed solver rules require a
different census.

For `m = N - 2` children and profile-group sizes `r`, the state count is
`sum(k=0..m)[B0(k) + 2*B1(k) + B2(k)] - B2(m)`, where
`Ba(k) = product_r binomial(ca(k) + r - 1, r)`.
The numbers of per-gene posterior classes are `c2=9`,
`c0=1,3,2k+2` for `k=0,1,>=2`, and `c1=3,7,2k+8` for the same ranges.
The branch census additionally weights supported outcomes and removes analytic
closures; runtime uses this branch count, not just the state count.

**Backend and necessary storage.** Under this census, the key needs
`2 + bit_length(N-2) + G*bit_length(17*(N-2)+13)` bits. The current native
accelerator supports at most 64 bits; wider keys use the exact Python backend.
That transition explains some large jumps between adjacent dimensions.
Native storage requires at least 48 bytes per retained state on the measured
build (40-byte node plus at least one eight-byte bucket per state). For the
captured 64-bit CPython layout, the Python lower bound per state is
`40 + 8*(G+2) + 56 + 24` bytes: key tuple, value/action pair, and dictionary
entry. It omits separate float objects, dictionary indices and spare capacity,
allocator overhead, and other solver storage. These are necessary bytes for
this representation, not universal lower bounds for all exact algorithms.

**Peak memory.** The native model uses observed resident-memory growth per
state, explicit bucket capacities, measured residual overhead, and a scenario
where old and new bucket allocations overlap. The median observed state-growth
rate is about 48.79 bytes per state before bucket storage. Bucket growth beyond
the observed capacities extrapolates the measured libc++ prime-growth rule.
The Python model subtracts key and dictionary storage from the two measured
runs, transfers the remaining bytes per state, adjusts tuple length for the
target gene count, and allows an extra dictionary table in the high scenario.
Different Python versions, allocators, builds, or table representations can
change these projections substantially.

Reapplying the native model to its calibration data gives 95.82 GB versus
96.03 GB measured for 20 people/five distinct genes, and 59.38 versus 59.59 GB
for ten people/six distinct genes. The central residual overhead transfers
poorly to small cases: ten people/four distinct genes project to 0.813 GB versus
0.210 GB measured. These comparisons reuse calibration data and are not
validation on independent, unseen large cases.

**Runtime.** `target_seconds = reference_seconds * target_branches /
reference_branches`. The reference must share the profile type and backend;
selection prioritizes matching people, then matching genes, then the largest
measured state count. The 15-person/seven-distinct-gene projection uses the
15-person/five-gene native run. The three Python crossing projections use the
matching-profile Python calibration above. There is no measured Python
15-gene or 15-person calibration, so those transfers are especially uncertain.
JSON includes sensitivity to other measured native throughputs when available;
it is not a fitted error interval. CPU/cache changes, varying branch work,
catalog setup, memory pressure, and paging can invalidate constant-throughputput
transfer. No cloud price is assumed; monetary cost would require a machine and
hourly rate as well as a credible duration on that machine.

## Verify exact results and measure a fresh run

Install the package as described in the [README](../README.md), then run:

```bash
python -m unittest discover -s checks -v
python benchmarks/verify.py --list
python benchmarks/verify.py --case nuclear_n10_g4_distinct --output benchmark-results/10p4g.json
python benchmarks/verify.py --case multigeneration_n5_g2_distinct --output benchmark-results/5p2g-general.json
```

The first benchmark took about four seconds with the measured native build;
the general-pedigree case took about three seconds. The verifier checks the
optimal root value and every root action value within absolute/relative
tolerance `1e-10`, accepts equal-valued optimal actions, and requires all four
work counters to match exactly. A mismatch or failed child process exits
nonzero. Timing and memory are measurements, not pass/fail thresholds.

To reproduce the Python calibration, or run all 18 reference cases:

```bash
python benchmarks/verify.py --case nuclear_n3_g10_alternating --case nuclear_n10_g4_distinct --backend python --output benchmark-results/python-calibration.json
python benchmarks/verify.py --all --output benchmark-results/all.json
```

The full suite takes many hours. These are **uncapped exact solves**; the
verifier imposes no state, transition, time, or memory budget. Existing output
files are rejected so a new run cannot overwrite evidence. Missing native
extensions cause exact Python fallback, potentially with much greater time and
memory; reports record the actual backend and source hashes.

Solver time ends when the result is available. Process wall time also includes
startup and natural teardown. Peak memory is the process resident high-water
mark sampled at result availability, before teardown; it is not a guaranteed
whole-process-lifetime peak. Memory is reported in bytes on macOS/Linux and as
unavailable on unsupported systems. Progress is emitted every 15 solver seconds.

The 34 software checks also exercise independent complete-world Bellman
enumeration on small cases. Agreement with recorded benchmark results alone
is a reproducibility check, not an independent proof of their optimality.
The [fresh verification report](../benchmarks/verification_20260915.json)
records successful reruns after a fresh package installation: ten people/four
distinct genes in 4.81 seconds with 0.201 GB peak memory, and the five-person,
two-gene multigeneration case in 3.06 seconds with 0.042 GB. Both processes
exited successfully and matched all checked values and work counts.
The current two-gene numerical compatibility evidence is documented in
[validation.md](validation.md).

The included data are portable extracts of the completed capacity study;
original result and artifact hashes preserve their provenance. External raw
logs and research experiments are not required to run these scripts. No
10–15-person/10–15-gene exact completion is claimed by these projections.
