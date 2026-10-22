# Input data and local preparation

The publicly distributed repository intentionally omits source data and previously computed reconstructions whose redistribution rights have not been established.

## Local private-file layout

After supplying datasets yourself, the following files should be present under the **repository root**:

```text
data/
  DataPhase28.dat
  Lvovsky_Data_Plots/
    rho_reconstruct_fock_nc_10.pkl
    CO_rho_reconstruct_pos.pkl
    homodyne_qst_cgan_rho.npy
  multimode_cat_plots/
    cat_state_data.dat
    plots/
      BI_N=10.npy
      BI_N=20.npy
      CO_N=10.npy
      CO_N=20.npy
      NN_N=10.npy
      NN_N=20.npy
      KQSE.npy                  # optional historical diagnostic
```

The historical large-cat data consist of 144,000 synthetic homodyne observations (72 phases, 2,000 outcomes per phase); the experimental kitten file contains 14,153 observations. These files are **not bundled**.

## Prepare your existing Google Drive ZIP exports

If you have permission to possess these data locally and have already downloaded the three archives, run this from the project root:

```powershell
python scripts/prepare_local_data.py --large-cat "C:\path\multimode_cat_plots.zip" --lvovsky "C:\path\Lvovsky_Data_Plots.zip" --physicality "C:\path\KQSE_physicality.zip"
```

All flags are optional, so you can import one archive at a time. The script unpacks only required input files and **never** alters the original archives. It does not unpack third-party source code. Check availability using `python scripts/check_local_inputs.py`.

**Security:** `.pkl` files may run arbitrary code when loaded. Only use trusted, reviewed inputs; do not run downloaded pickle data from unknown sources.

## Scientific provenance

The kitten measurements originate in the historical optical kitten-state experiment (see A. I. Lvovsky and J. Mlynek, *Phys. Rev. Lett.* **88**, 250401, 2002). Saved competitor matrices used in the comparisons originate from prior BI/CO/CGAN reconstructions associated with related work; their redistribution permission has not been independently established. The source of each externally supplied array should be documented before public redistribution.

The numerically generated benchmark tables and manuscript figures under `results/` are summary outputs, not substitutable for the underlying raw inputs or baseline matrices.
