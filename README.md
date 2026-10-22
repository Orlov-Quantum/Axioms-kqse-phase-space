# Kernel Quantum State Estimation (KQSE)

**Companion numerical code** for *Optimal Nonparametric Estimation of Phase-Space Representations for Non-Gaussian Continuous Variable Quantum States* by V. Orlov and L. Markovich.

This repository documents **Kernel Quantum State Estimation (KQSE)** from homodyne measurements. A reusable estimate of the tomographic characteristic function is transformed to the **Wigner function** and **photon-number tomograms**. The paper presents theoretical error bounds, numerical examples with non-Gaussian states, and diagnostics of quantum-state physicality.

## Contents

| File | Numerical experiment |
| --- | --- |
| [`notebooks/01_small_cat_and_scaling.ipynb`](notebooks/01_small_cat_and_scaling.ipynb) | Three-component coherent cat, amplitude `a = 1+0.5j`; KQSE vs GMM-3; `n = 500, 1000, 2000`, `R = 100` |
| [`notebooks/02_large_cat_comparison.ipynb`](notebooks/02_large_cat_comparison.ipynb) | Larger coherent cat, `a = 3+0.5j`; KQSE vs saved BI, CO and CGAN reconstructions, two Fock cutoffs |
| [`notebooks/03_experimental_kitten.ipynb`](notebooks/03_experimental_kitten.ipynb) | Experimental homodyne kitten data; KQSE with and without phenomenological noise correction; comparisons with GMM-3, BI, CO, CGAN |
| [`scripts/kqse_physicality_all_in_one.py`](scripts/kqse_physicality_all_in_one.py) | Photon-number negativity, finite Bochner matrices, angular-binning tests and Monte Carlo controls |

Precomputed **aggregate numerical tables** and the article figures are in [`results/`](results/). These do not include the underlying experimental measurements or baseline density matrices.

## Installation

Use Python 3.11 or newer and a virtual environment if possible:

```powershell
python -m pip install -r requirements.txt
```

Run Jupyter **from the repository root**:

```powershell
python -m jupyter lab
```

Open one of the three notebooks in `notebooks/` and select a Python kernel.
The small-cat simulation can generate its own synthetic data; its full `R=100` run may take considerable time and writes resumable checkpoints into a locally ignored output directory.

## Reproducing the paper

- **Small synthetic cat / scaling:** no external input is needed. The finalized article values are provided in `results/tables/synthetic_errors_R100.csv`. The full independent `R=100` rerun has not yet been performed as a release test.
- **Large synthetic cat:** the comparison requires the original 144,000 synthetic homodyne observations and saved BI/CO/CGAN matrices (not bundled).
- **Experimental kitten:** requires the experimental observations and the saved third-party baseline arrays (not bundled).
- **Physicality:** requires the same experimental observation file. After obtaining it, run:

```powershell
python scripts/kqse_physicality_all_in_one.py --data data/DataPhase28.dat --output outputs/physicality_full
```

Do not use `--quick` when comparing Monte Carlo counts with the article. Details and expected numerical checks: [Reproducibility](docs/REPRODUCIBILITY.md), [Data preparation](docs/DATA.md).

## Methodological note

The numerical computations employ a data-dependent bandwidth approximation replacing the exact logarithmic oracle expression by its leading small-argument expansion, along with the stated numerical clipping rules. The theoretical and numerical bandwidths should not be conflated; see the manuscript and the reproducibility notes.

The finite synthetic/experimental comparisons of BI, CO and CGAN use **previously computed reconstructions**, rather than rerunning these algorithms. The repository does not claim to contain their original implementations. See [third-party provenance](docs/THIRD_PARTY.md).

## Citation and license

Please cite the accompanying manuscript (complete bibliographic information will be added after publication). The original code provided here is released under the repository's MIT license. Third-party software, baseline arrays and experimental data are **not** relicensed by it.
