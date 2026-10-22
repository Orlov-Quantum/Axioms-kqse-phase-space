#!/usr/bin/env python3
"""Report file presence without opening or deserializing inputs."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
required = {
  'experimental homodyne': ['data/DataPhase28.dat'],
  'experimental BI / CO / CGAN': [
    'data/Lvovsky_Data_Plots/rho_reconstruct_fock_nc_10.pkl',
    'data/Lvovsky_Data_Plots/CO_rho_reconstruct_pos.pkl',
    'data/Lvovsky_Data_Plots/homodyne_qst_cgan_rho.npy',
  ],
  'large cat': ['data/multimode_cat_plots/cat_state_data.dat'] + [
    f'data/multimode_cat_plots/plots/{m}_N={n}.npy'
    for m in ('BI', 'CO', 'NN') for n in (10, 20)
  ],
}
for group, files in required.items():
    missing=[f for f in files if not (ROOT/f).is_file()]
    print(('READY' if not missing else 'MISSING') + ': ' + group)
    for f in missing: print('  - ' + f)
