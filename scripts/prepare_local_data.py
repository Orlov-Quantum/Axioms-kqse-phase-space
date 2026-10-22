#!/usr/bin/env python3
"""Extract only whitelisted local inputs from the user's OWN trusted ZIP exports.

This is a local convenience tool; input data are not distributed or committed.
"""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def put(zf, suffix, dest):
    matches = [n for n in zf.namelist() if n.endswith(suffix) and not n.endswith('/')]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {suffix!r}, found {len(matches)}")
    path = ROOT / 'data' / dest
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print('Already present (not overwritten):', path.relative_to(ROOT))
        return
    with zf.open(matches[0]) as src, path.open('wb') as dst:
        import shutil
        shutil.copyfileobj(src, dst)
    print('Installed local input:', path.relative_to(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--large-cat', type=Path, help='ZIP export of multimode_cat_plots')
    p.add_argument('--lvovsky', type=Path, help='ZIP export of Lvovsky_Data_Plots')
    p.add_argument('--physicality', type=Path, help='KQSE physicality archive containing DataPhase28.dat')
    args = p.parse_args()
    if not (args.large_cat or args.lvovsky or args.physicality):
        p.print_help()
        return
    if args.large_cat:
        with zipfile.ZipFile(args.large_cat) as zf:
            put(zf, '/cat_state_data.dat', 'multimode_cat_plots/cat_state_data.dat')
            for meth in ('BI','CO','NN'):
                for cutoff in (10,20):
                    name=f'{meth}_N={cutoff}.npy'
                    put(zf, '/plots/'+name, 'multimode_cat_plots/plots/'+name)
            if any(n.endswith('/plots/KQSE.npy') for n in zf.namelist()):
                put(zf, '/plots/KQSE.npy', 'multimode_cat_plots/plots/KQSE.npy')
    if args.lvovsky:
        with zipfile.ZipFile(args.lvovsky) as zf:
            for name in ('rho_reconstruct_fock_nc_10.pkl','CO_rho_reconstruct_pos.pkl','homodyne_qst_cgan_rho.npy'):
                put(zf, '/'+name, 'Lvovsky_Data_Plots/'+name)
    if args.physicality:
        with zipfile.ZipFile(args.physicality) as zf:
            put(zf, 'data/DataPhase28.dat', 'DataPhase28.dat')
    print('Done. Files are intended for local use only (ignored by Git).')


if __name__ == '__main__':
    main()
