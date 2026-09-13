# Package verification

Verified on 2026-09-13 with Python 3.13.5 and NumPy 2.3.4.

- The inference and nuclear-family numerical implementations match the source
  after the documented import and public-name changes. The junction-tree code
  matches after removing the unused separator-partition definitions and exports.
  Comparison used Python syntax trees with documentation removed.
- All four standalone software checks pass. They cover independent per-gene
  enumeration on a five-person, two-gene case; joint marginal probabilities;
  the one-test expected-risk identity; legal complete panels; initial evidence;
  impossible evidence; panel costs; and STOP ties.
- Target-size checks cover 18 combinations: 10, 12, and 15 people crossed with
  10, 12, and 15 genes, for nuclear and multigeneration pedigrees. Both policies
  terminate legally. The generic and specialized nuclear calculations agree
  at the checked observed histories within numerical tolerance.
- A wheel was built and installed into a separate directory. From outside the
  source repository, that installation passes the same four checks and runs
  all three supplied JSON examples with both policies. No `bayes_risk_audit`
  or `genetic_dp` modules were imported.

To repeat the software checks after installation:

```bash
python -m unittest discover -s checks -v
```

These checks establish extraction correctness and example execution. They do
not estimate policy quality, speed relative to other implementations, or
performance across arbitrary pedigree topologies. No research results are
distributed with this package. Other supported Python versions were not tested
in this packaging pass.
