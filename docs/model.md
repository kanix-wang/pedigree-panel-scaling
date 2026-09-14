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
`gt_code2`. It evaluates every legal test action and every supported panel
outcome and memoizes values by the complete observation state. The maximizing
action can depend on all observed genes: the policy is not factored into
separate per-gene policies. Exact ties select STOP first, then individuals in
their supplied order, using strict floating-point comparisons.

When only one individual remains, every post-test continuation value is zero.
The solver therefore closes that test action analytically with value $R_i(O)$,
without enumerating terminal panels. This is an exact simplification of the
recurrence, not a shortened planning horizon.

A completed `ExactSolution` contains the root optimum, maximizing root action,
all root action values, and the solved value/action maps. `value_at` and
`action_at` query those maps for supported observation states reachable from the
solved root; supported fully tested successors return zero and STOP.

The defaults in `ExactLimits` are 100,000 entered states, 2,000,000 enumerated
transitions, and 60 solver seconds. The state counter counts entered decision
states rather than a hypothetical enumeration of analytically closed terminal
panels. `terminal_actions_closed` counts those last-person test actions.
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
completed checks and measured limits.
