\
# Numerical reproducibility

The checked code consists of three Jupyter notebooks and one standalone Python script. In the publication copies, existing Jupyter outputs were removed. **All numerical cell sources have been preserved**, with one environment/path configuration cell edited in each of the large-cat and experimental-kitten notebooks. The small-cat code and physicality CLI are unchanged.

| Study | Validation carried out on the source material (20 Oct 2026) | Public clone alone |
| --- | --- | --- |
| Small coherent cat, `a=1+0.5j`, and scaling | Final `R=100` saved results matched the manuscript: 60/60 MISE and gridwise MSE values at printed precision. Full Monte Carlo rerun still outstanding. | Synthetic run possible, but computationally expensive |
| Large coherent cat, `a=3+0.5j` | Recomputed from supplied original data and saved BI/CO/CGAN estimates: 80/80 ISE and squared maximum-discrepancy values matched saved final table (about `1.2e-16` maximum deviation). | Requires privately obtained large-cat input and baseline reconstructions |
| Experimental kitten | Recomputed from 14,153 quadrature data records and saved BI/CO/CGAN reconstructions: 60/60 final comparison values matched (about `1.1e-16` maximum deviation). | Requires privately obtained raw data and baseline reconstructions |
| Physicality diagnostics | Full script rerun passed 38/38 internal reference checks, including 150-realization kitten controls and 180-realization six-state controls. | Requires privately obtained experimental data |

**Caution:** agreement with stored benchmark outputs and a rerun with the same third-party arrays is not the same as retraining BI, CO, and CGAN. PDF figures were matched by filenames; not independently checked for pixel-identical rendering across environments.

## Implementation fidelity

Original mathematical and numerical cells were not refactored or reordered. The historical notebook sections remain intact, including diagnostics not essential to the article. Paths changed only in the initial setup cells; the precise changes are documented in `audit/CODE_CHANGES.md`. Running a notebook from a different working directory may require following `docs/DATA.md`.

The practical Gaussian bandwidth is an approximation to the logarithmic rule appearing in the oracle discussion:

\[\sqrt{\log(1+x)} \approx \sqrt{x}\quad \text{for small }x.\]

The pilot-CF modulus floor and bandwidth clipping differ by numerical experiment and are **not silently unified**. In particular, the experimental kitten uses no floor on the pilot-CF modulus, while the small/large synthetic cat benchmarks use `phi_abs_floor = 0.03`.

The quantum-Bochner and photon-negativity tests are **diagnostics**, not proof that an estimated characteristic function corresponds to a physical quantum state.

## Commands

```powershell
python -m pip install -r requirements.txt
python scripts/check_local_inputs.py
python -m jupyter lab
```

For the physicality calculation, run:

```powershell
python scripts/kqse_physicality_all_in_one.py --data data/DataPhase28.dat --output outputs/physicality_full
```

Avoid `--quick` when testing Monte Carlo counts quoted in the manuscript.
