# Verification record — 20 October 2026

This record summarizes checks completed **before assembling this public package** on private input files. It is not a guarantee that each notebook has been run from a newly cloned public repository.

- Small cat R=100: saved output and archived `synthetic_errors_R100.csv` match the manuscript tables 60/60 at printed precision. Full rerun of all 100 repeats remains to be done.
- Large cat: numerical reconstruction with source synthetic data and fixed BI, CO and CGAN matrices reproduced the complete final comparison table: 80/80 values to approximately `1.2e-16`.
- Experimental kitten: a complete recomputation with the original homodyne data and the provided fixed BI, CO and CGAN arrays reproduced all 60 comparison values to approximately `1.1e-16`.
- Physicality script: the full calculation completed with 38/38 validation checks. It also reproduced the reported Monte Carlo extreme counts.

None of these private raw inputs or third-party reconstruction matrices is distributed here. The article figures and aggregate tables in `results/` permit inspection without those inputs.
