# Model and scaling

The unit of a test is one individual's complete panel. A case describes one
family, a shared list of genes, inheritance relationships, and numerical reward
and cost parameters. The scaling target contains 10–15 individuals and 10–15
genes. Current-state inference and full sequential optimization have different
computational costs; exact optimization at that target has not been demonstrated.

## Inheritance and observations

For gene $g$ and individual $i$, the genotype $X_{gi}\in\{0,1,2\}$ counts copies
of the modeled allele. With allele frequency $f_g$, a founder has probabilities

$$
\Pr(X_{gi}=0)=(1-f_g)^2,\qquad
\Pr(X_{gi}=1)=2f_g(1-f_g),\qquad
\Pr(X_{gi}=2)=f_g^2.
$$

Founders are independent in this model. For a child whose parents have genotypes
$x$ and $y$, put $u=x/2$ and $v=y/2$. Mendelian transmission gives

$$
\Pr(X_{g,\mathrm{child}}=0\mid x,y)=(1-u)(1-v),
$$

$$
\Pr(X_{g,\mathrm{child}}=1\mid x,y)=u(1-v)+(1-u)v,\qquad
\Pr(X_{g,\mathrm{child}}=2\mid x,y)=uv.
$$

Genes are independent, including after conditioning on observed complete
panels. Relatives within a gene remain dependent. The model does not include
linkage or observation error.

The observation array has shape `(genes, people)`. An untested column contains
only `-1`; a tested column contains only `0`, `1`, or `2`. Partial panels are not
public states. Initial evidence is preserved, repeat tests are rejected, and
inference rejects evidence with zero probability under the model. A simulated
trajectory draws a coherent family genotype configuration conditional on its
initial evidence and reveals panels only as the policy requests them.

## Risk, rewards, and costs

Given current observations $O$, let $U$ be the set of untested individuals and
let $p_{gi}=\Pr(X_{gi}>0\mid O)$ be the carrier probability. Input maps
`a_gene`, `b_gene`, and `omega_gene` are multiplied by two for every nonfounder
(any individual listed as a child), preserving the extracted model's convention.
The `delta_gene` map is unchanged. The resulting coefficients are $a_{gi}$,
$b_{gi}$, $\omega_{gi}$, and $\delta_{gi}$.

Define the nonnegative risk weight and affine reward term by

$$
\lambda_{gi}=-(a_{gi}\delta_{gi}+b_{gi}),\qquad
A_{gi}=a_{gi}(1-\delta_{gi})p_{gi}+\omega_{gi}.
$$

The current residual risk, baseline, and stop reward are

$$
\Psi(O)=\sum_{i\in U}\sum_g\lambda_{gi}p_{gi}(1-p_{gi}),\qquad
H(O)=\sum_{i\in U}\sum_g A_{gi},\qquad
S(O)=H(O)-\Psi(O).
$$

For an untested individual $i$, the expected complete panel cost is

$$
C_i(O)=c_{\mathrm{fixed}}+
c_{\mathrm{variable}}\left[1-\prod_g(1-p_{gi})\right].
$$

Thus the variable component applies when at least one gene is positive. The
product uses independence across genes. The immediate expected test reward is
$R_i(O)=\sum_g A_{gi}-C_i(O)$.

## Exact sequential optimization

Let $\mathcal{X}_i(O)$ contain the complete genotype panels with positive
conditional probability for untested individual $i$. Independence across genes
gives the probability of a particular panel $x$:

$$
P_i(x\mid O)=\prod_g\Pr(X_{gi}=x_g\mid O).
$$

The full-horizon Bellman recurrence is

$$
V(O)=\max\left\{S(O),\ \max_{i\in U}\left[
R_i(O)+\sum_{x\in\mathcal{X}_i(O)}P_i(x\mid O)V(O\cup x_i)
\right]\right\}.
$$

When no untested individuals remain, $V(O)=0$. `solve_exact` implements this
unrestricted recurrence, matching the backward-induction formulation in
`gt_code2`. It uses exact state equivalences and analytical closures described
below. For actions requiring expansion, it enumerates every supported complete
panel outcome. The maximizing action can depend on all observed genes: the
policy is not factored into separate per-gene policies. Exact ties select STOP
first, then individuals in their supplied order, using strict floating-point
comparisons.

### Independent remaining decisions

When the remaining individuals are conditionally independent given the evidence,
testing one does not change another's posterior or reward. Define each
individual's contribution to stopping as

$$
S_i(O)=\sum_g\left[A_{gi}-\lambda_{gi}p_{gi}(1-p_{gi})\right].
$$

The complete optimal value and every test action value then follow directly:

$$
V(O)=\sum_{i\in U}\max\{S_i(O),R_i(O)\},\qquad
Q_i(O)=R_i(O)+\sum_{j\in U\setminus\{i\}}\max\{S_j(O),R_j(O)\}.
$$

The STOP action retains value $S(O)=\sum_i S_i(O)$. These formulas retain the
shared cost of each complete panel and allow every subset of individuals to be
tested before stopping. The solver applies them under a sufficient condition:
at each gene, every edge of the moralized pedigree graph has an endpoint whose
posterior genotype support contains exactly one value. A moralized graph joins
each parent to its child and joins co-parents. Fixing those constant variables
leaves no dependence edges between uncertain variables. The check uses support
size, not a rounded probability or a threshold near one.

The last-person case is included: all post-test continuation values are zero,
so that test action has value $R_i(O)$ without terminal panel enumeration. Neither
closure restricts the planning horizon.

### Exact nuclear-family state compression

For a nuclear family, the decision state can be represented by the posterior
over the nine ordered parental genotype pairs at each gene, flags recording
which parents have been tested, and the number of untested children. Children
share inheritance and reward/cost profiles, so their identities do not affect
future values. An equivalent child action is evaluated once, and its action
value is assigned to every legal child action. A stored child action is decoded
to the first untested child in the supplied individual order.

The state key represents the parental posterior without rounding probabilities.
For each gene, observed child genotype counts determine a nine-cell likelihood
vector. Positive Mendelian likelihood factors belong to $\{1/4,1/2,1\}$, so
their base-two logarithms and their accumulated sums are integers. The key
retains the support mask, including parental evidence and founder-prior support,
and subtracts the largest supported log likelihood from all supported cells.
These normalized integer exponents identify proportional likelihood vectors.
Within a fixed gene profile, the parental prior is fixed, so equality of these
vectors identifies the same posterior exactly.

Gene rows are sorted only within groups having exactly equal numerical allele
frequencies and all effective `a`, `b`, `delta`, and `omega` coefficients.
Permuting these genes preserves inheritance, rewards, and the complete-panel
cost. Gene profiles are not rounded or approximately matched. Tested-parent
flags remain part of the state even when an untested parent's genotype has
already been inferred with certainty, because testing eligibility still differs.

The nuclear implementation interns each gene profile's signatures and caches
its genotype probabilities, rewards, and single-gene transitions. The Bellman
loop combines these cached transitions into complete panel outcomes. The optional
C++ accelerator executes the same recurrence with packed integer keys; it uses
the Python implementation when the complete key cannot fit in 64 bits. That
representation choice does not impose a gene-count or search-state limit.

General pedigrees use complete observation arrays as memoization keys. Both
representations preserve the full Bellman recurrence. There is no probability
rounding, supported-outcome pruning, sampling, or horizon restriction.

### Queries, progress, and optional limits

A completed `ExactSolution` contains the root optimum, maximizing root action,
all legal root action values, and the canonical value/action maps. `value_at`
and `action_at` accept supported descendants of the solved root and validate
that initial evidence and root observations are preserved. They reuse a
canonical memo entry or evaluate an analytically closed descendant exactly;
they do not start a new search. Fully tested successors return zero and STOP.

All `ExactLimits` fields default to `None`: there is no default cap on states,
transitions, or solver time. Limits apply only when explicitly supplied. The
reported counters measure the work of the reduced exact search:

- `states_evaluated` counts entered canonical decision states, excluding memo
  hits. It is not a count of all distinct observation histories represented.
- `transitions_evaluated` counts supported panel outcomes actually enumerated
  for representative actions. Analytically closed actions contribute none.
- `terminal_actions_closed` counts analytically closed last-person test actions.
- `independent_states_closed` counts states closed with more than one untested
  individual under the independence condition.
- `elapsed_seconds` measures solver time, excluding inference-engine construction.

An optional `progress` callback receives an `ExactProgress` snapshot with these
fields at most once per second. The completed solution carries final counters;
a progress snapshot supplies no provisional optimum. The solver itself prints
nothing, and callback exceptions propagate. CLI `--progress` writes JSON
snapshots to standard error, leaving the final result on standard output.

Time checks occur between operations and exclude inference-engine construction;
they are cooperative rather than a hard process deadline.

An exceeded limit raises `ExactLimitExceeded` with counters and the stopping
reason. The command line reports `resource_limit`, returns `null` for the
optimal value and action, and exits with code 2. An incomplete solve returns no
exact solution and supplies no greedy substitute.

## Greedy baselines

`recursive_myopic` chooses an individual whose immediate reward improves on the
current stop reward, or stops if none does. It applies the same rule after each
new panel; it does not search future decision sequences.

`information_greedy` chooses the largest positive one-test gain

$$
G_i(O)=\Psi(O)-\mathbb{E}\!\left[\Psi(O\cup X_i)\mid O\right]-C_i(O),
$$

and otherwise stops. Here $X_i$ is the complete panel and testing removes $i$
from the untested set. Its information gain includes updated risks for relatives.
Per-gene conditional moments compute this expectation without enumerating all
$3^{\lvert\mathrm{genes}\rvert}$ possible complete panels. Both policies use
deterministic tie handling based on the supplied individual order.

Recursive myopic starts with STOP and replaces the current best action only
when its reward improves by more than `1e-6`. Information greedy considers
STOP first, then individuals in order; it uses absolute and relative
tolerances of `1e-10` on the scores `S(O) + G_i(O)`.

## Inference scaling and optimization limits

The general `PedigreeInference` engine uses exact sum-product propagation on a
junction tree and batches genes in NumPy. A clique of $k$ individuals has $3^k$
genotype assignments per gene, so the largest intermediate factor depends on
pedigree graph width. Increasing the number of independent genes adds batched
work instead of creating a joint cross-gene state table. Optional pairwise
quantities for one-test information gains add work across individuals.

For a single nuclear family, `NuclearInference` conditions on the nine possible
parental genotype pairs at each gene. Observed children's genotype counts update
that parental posterior. Conditional on their parents, unobserved children are
independent. This specialization produces the same current-state posterior
quantities as the general engine on its supported family structure.

These inference savings do not remove the joint outcome branching of Bellman
optimization. A single test can have up to $3^{\lvert\mathrm{genes}\rvert}$
supported panels, and future decisions depend on the resulting complete
observation state. Low pedigree width alone does not make that decision graph
small.

Exactness refers to the stated probabilistic model and floating-point
arithmetic. A completed exact solve establishes the optimal sequential policy
on its solved observation graph. A resource-limited attempt establishes no
optimum, and executing a greedy trajectory at 10–15 individuals and 10–15 genes
does not establish exact optimization at that size. The default CLI uses a
three-person, two-gene case; [the validation record](validation.md) records
completed ten-person four- and five-gene examples, numerical compatibility, and
the remaining whole-table memory limitations at larger gene counts.
