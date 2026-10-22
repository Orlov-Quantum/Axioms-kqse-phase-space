#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KQSE photon-tomogram physicality: complete reproducible numerical audit.

Based on the original experiment, optical kitten control simulations and the
six-state synthetic study, with all computation routines gathered in ONE file.
Only numpy, scipy, pandas (and matplotlib for optional figures) are required.

INPUT: DataPhase28.dat (original homodyne data). An existing reconstructed
       arrays .npz may be supplied as an INDEPENDENT optional checkpoint.
OUTPUT: CSV tables, source arrays, validation report, and numerical graphs.

Usage:
  python kqse_physicality_all_in_one.py --data DataPhase28.dat --output run
  python kqse_physicality_all_in_one.py --data DataPhase28.dat --output run --quick
  python kqse_physicality_all_in_one.py --data DataPhase28.dat --output run --checkpoint KQSE_kitten_recovered_arrays.npz

The full run uses 150 Monte Carlo realizations per kitten scenario and
180 realizations per synthetic state and bin count. Random seeds are fixed.
Sampling and noise models are APPROXIMATIONS and are not a calibrated model
of the historical optical experiment. Full quantum physicality cannot be
certified by finitely many tests. Finite-r sums are not complete normalization.

This script bundles calculations, not novel physics: the empirical and KCFE
estimators, photon kernels and quantum Bochner matrix implement equations in
our manuscript. Kept original experimental phase-folding convention for
exact numeric backward compatibility with the historical notebook.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, csv, time, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import erf, erfinv, ndtr, eval_laguerre
from numpy.polynomial.legendre import leggauss

KAPPA = 0.9
R_LIST = (0,1,2,3)
TWO_PI = 2.0 * np.pi
VARIANTS = (
    ('model', 'phi_model', 'Approximate kitten reference'),
    ('nocorr', 'phi_exp_kqse_nocorr', 'KQSE without noise correction'),
    ('corr', 'phi_exp_kqse_corr', 'KQSE with noise correction (kappa=0.9)'),
)
p=.6810463
amp=.3
refl2=.925
c0=np.sqrt(1-refl2)/np.sqrt(1-refl2+amp*amp)
c1=amp/np.sqrt(1-refl2+amp*amp)
A=p*c1*c1
B=np.sqrt(2)*p*c0*c1
sqrt_pi=np.sqrt(np.pi)
STATES=('vacuum','coherent','thermal','fock1','squeezed','kitten')


# ======================================================================
# EXPERIMENTAL GRID, CF ESTIMATORS, ALTERNATIVE PHASE TREATMENTS
# Extracted from kqse_phase_methods.py; all helpers self-contained below.
# ======================================================================
def original_phase_convention(theta, x):
    """Preserve the original notebook's phase folding and sign convention EXACTLY."""
    k=np.floor(theta/np.pi).astype(int)
    folded=np.mod(theta,np.pi)
    return folded, x*np.where(k%2==0,-1.,1.)


def cart_grid_geometry(mu,nu):
    MU,NU=np.meshgrid(mu,nu,indexing='ij')
    radius=np.hypot(MU,NU)
    ang=np.mod(np.arctan2(NU,MU),2*np.pi)
    return radius,np.mod(ang,np.pi),np.where(ang>=np.pi,-1.,1.)


def sanitize_cf(cf, mu,nu):
    cf=(cf+cf[::-1,::-1].conj())/2
    cf[np.argmin(abs(mu)),np.argmin(abs(nu))]=1.+0j
    return cf


def cf_from_binned(theta, x, mu,nu, N, correction, pilot='full_line'):
    """Same nearest phase bin and h in [0.01,0.5] as original, full-line KDE pilot.
    Full-line pilot validated independently against old published 20-bin calculation.
    """
    th,xf=original_phase_convention(theta,x)
    bi=np.minimum(np.floor(th/(np.pi/N)).astype(int),N-1)
    samples=[xf[bi==j] for j in range(N)]
    if min(map(len,samples))<3:raise ValueError('Too few samples per bin')
    radius,theta0,sgn=cart_grid_geometry(mu,nu)
    binids=np.minimum(np.floor(theta0/(np.pi/N)).astype(int),N-1)
    flat_r=radius.ravel(); flat_b=binids.ravel(); flat_s=sgn.ravel()
    phi=np.ones(flat_r.shape,dtype=np.complex128); bw=np.zeros(len(phi))
    for b in range(N):
        inds=np.where((flat_b==b)&(flat_r>1e-10))[0]
        if not len(inds):continue
        s=samples[b]; n=len(s)
        t=flat_r[inds]/(KAPPA if correction else 1.)
        raw=np.exp(1j*t[:,None]*s[None,:]).mean(axis=1)
        sigma=np.std(s,ddof=1)*(3*n/4)**(-.2)
        pilot=np.abs(raw)*np.exp(-.5*(sigma*t)**2)
        pilot=np.clip(pilot,np.finfo(float).tiny,1.)
        h=np.clip((2./t)*np.sqrt(np.maximum(0.,1.-pilot**2)/(n*pilot**2)),.01,.5)
        val=raw*np.exp(-.25*(t*h)**2)
        if correction:val*=np.exp(.5*((1-KAPPA)/KAPPA*flat_r[inds])**2)
        phi[inds]=np.where(flat_s[inds]>0,val,val.conj()); bw[inds]=h
    return sanitize_cf(phi.reshape(len(mu),len(nu)),mu,nu),bw.reshape(len(mu),len(nu))


def angle_window_signed(theta_fold, Xfold, target, bandwidth):
    # Antiperiodic quadrature when folding angle by pi.
    d0=theta_fold-target
    d=(d0+np.pi/2)%np.pi-np.pi/2
    mask=abs(d)<=3*bandwidth
    wrap=np.rint((d0[mask]-d[mask])/np.pi).astype(int)
    x=Xfold[mask]*np.where(wrap%2==0,1.,-1.)
    weight=np.exp(-.5*(d[mask]/bandwidth)**2)
    return x,weight


def cf_phase_local(theta,x,mu,nu,bandwidth_deg,correction, min_neff=20):
    """Phase-local Nadaraya-Watson CF with weighted Silverman bandwidth pilot.
    Data are independent shots at recorded phases, and same data are reused for
    query frequencies. This estimator is exploratory, not covered by Theorem 4.1.
    """
    th,xf=original_phase_convention(theta,x)
    radius,ang,sgn=cart_grid_geometry(mu,nu)
    out=np.ones_like(radius,dtype=complex); n_eff=np.zeros_like(radius)
    w_kcfe=np.zeros_like(radius)
    b=np.deg2rad(bandwidth_deg)
    # Half-plane only: exact Hermitian symmetry is then automatic.
    for i,j in zip(*np.where((sgn>0)&(radius>1e-9))):
        xx,ww=angle_window_signed(th,xf,ang[i,j],b)
        sw=ww.sum();ww=ww/sw
        neff=1./np.dot(ww,ww)
        n_eff[i,j]=neff
        if neff<min_neff:raise RuntimeError('Too few effective samples: '+str(neff))
        t=radius[i,j]/(KAPPA if correction else 1.)
        z=np.dot(ww,np.exp(1j*t*xx))
        sm=np.dot(ww,xx);std=np.sqrt(np.dot(ww,(xx-sm)**2)*(neff/(neff-1)))
        sig=std*(3*neff/4)**(-.2)
        phip=np.clip(abs(z)*np.exp(-.5*(sig*t)**2),np.finfo(float).tiny,1.)
        h=np.clip(2./t*np.sqrt(max(0,1-phip**2)/(neff*phip**2)),.01,.5)
        result=z*np.exp(-.25*(t*h)**2)
        if correction:result*=np.exp(.5*((1-KAPPA)/KAPPA*radius[i,j])**2)
        out[i,j]=result
        w_kcfe[i,j]=h
    # Mirror direct positive-half-plane results for negative plane; some boundary
    # grid points satisfy phase 0 exactly, their opposite sits in other half.
    phi=sanitize_cf(out,mu,nu)
    # Above sanitize_cf would halve values of the positive half-plane because
    # negative half contains ones. Explicitly copy values to conjugate locations.
    good=(sgn>0)&(radius>1e-9)
    ii,jj=np.where(good)
    phi[ii,jj]=out[ii,jj]
    phi[len(mu)-1-ii,len(nu)-1-jj]=np.conj(out[ii,jj])
    phi[np.argmin(abs(mu)),np.argmin(abs(nu))]=1.+0j
    return phi,n_eff,w_kcfe


def photon_kernel_mat(mu,nu,alpha,rs=R_LIST):
    MU,NU=np.meshgrid(mu,nu,indexing='ij')
    dmu=mu[1]-mu[0];dnu=nu[1]-nu[0]
    u=(MU*MU+NU*NU).ravel()/2
    exp0=np.exp(-u/2)
    phase=np.exp(1j*np.sqrt(2)*np.outer(alpha,MU.ravel()))
    return np.concatenate([phase*(exp0*eval_laguerre(r,u))[None,:]*(dmu*dnu/(2*np.pi)) for r in rs],axis=0)


def photon_from_phi(cf,KM,n_alpha=121,nr=4):
    w=(KM@cf.ravel()).reshape(nr,n_alpha)
    if np.max(np.abs(w.imag))>1e-8:raise ValueError('Real-valued photon tomogram expected')
    return w.real


def bochner_eigen(cf,mu,nu):
    d=2*(mu[1]-mu[0]);
    pts=np.array([(0,0),(d,0),(-d,0),(0,d),(0,-d),(d,d),(d,-d),(-d,d),(-d,-d)])
    origin=len(mu)//2;step=mu[1]-mu[0]
    B=np.empty((len(pts),len(pts)),complex)
    for i,(a,b) in enumerate(pts):
        for j,(c,e) in enumerate(pts):
            ii=int(round((c-a)/step))+origin;jj=int(round((e-b)/step))+origin
            B[i,j]=np.exp(.5j*(a*e-b*c))*cf[ii,jj]
    B=(B+B.conj().T)/2
    return float(np.linalg.eigvalsh(B)[0])


def compute_metrics(name,method,cf,KM,archive,extra=None):
    alpha=archive['exp_alpha_eval'];mu=archive['exp_mu_grid'];nu=archive['exp_nu_grid'];
    w=photon_from_phi(cf,KM,n_alpha=len(alpha))
    wm=np.stack([archive[f'w_model_r{r}'].real for r in R_LIST])
    neg=np.maximum(-w,0);pos=np.maximum(w,0)
    row={'method':method,'setting':name,'bochner_min':bochner_eigen(cf,mu,nu),
         'neg_mass_max_r0_3':float(neg.sum(axis=0).max()),
         'neg_mass_integral_r0_3':float(np.trapezoid(neg.sum(axis=0),x=alpha)),
         'partial_sum_max_r0_3':float(w.sum(axis=0).max()),
         'negative_fraction_of_positive_integrals_r0_3':float(np.trapezoid(neg.sum(axis=0),x=alpha)/np.trapezoid(pos.sum(axis=0),x=alpha))}
    for r in R_LIST:
        row[f'min_r{r}']=float(w[r].min());row[f'neg_nodes_r{r}']=int((w[r]<-1.e-8).sum());
        row[f'ISE_r{r}']=float(np.sum(abs(w[r]-wm[r])**2)*(alpha[1]-alpha[0]))
    if extra:row.update(extra)
    return row,w


def direct_polar(theta,x,alpha,rs=R_LIST,cut=6.5,radial_nodes=46,damping=.2,chunk=1200,correction=False):
    """Direct polar inversion with each actual folded phase, no angular binning.
    Square Fourier cutoff retained: |t cos(theta)|<=cut, |t sin(theta)|<=cut.
    Constant Gaussian radial filter exp(-damping^2*t^2/4) regularizes frequency.
    Assumes approximately uniform phase sampling. Not the same KQSE estimator.
    """
    theta,X=original_phase_convention(theta,x)
    nodes,weights=leggauss(radial_nodes)
    n=len(x);results=np.zeros((len(rs),len(alpha)))
    for begin in range(0,n,chunk):
        end=min(n,begin+chunk)
        th=theta[begin:end];xx=X[begin:end];co=np.cos(th)
        tmax=cut/np.maximum(abs(np.cos(th)),abs(np.sin(th)))
        rad=.5*(nodes[None,:]+1)*tmax[:,None]
        integ=.5*tmax[:,None]*weights[None,:]
        gain = np.exp(.5*(((1-KAPPA)/KAPPA)*rad)**2) if correction else 1.0
        factor=integ*rad*np.exp(-.25*(1+damping**2)*rad**2)*gain
        ell=rad**2/2
        # integrate phase, batch alpha to constrain memory
        for aj in range(0,len(alpha),16):
            ae=alpha[aj:aj+16]
            z=(xx[:,None]/(KAPPA if correction else 1.0)+np.sqrt(2)*co[:,None]*ae[None,:])
            # shape sample x alpha x nodes
            c=np.cos(z[:,:,None]*rad[:,None,:])
            for ir,r in enumerate(rs):
                pref=factor*eval_laguerre(r,ell)
                results[ir,aj:aj+len(ae)]+=np.einsum('san,sn->a',c,pref,optimize=True)
    return results/n


def direct_model_polar(alpha,rs,cut=6.5,radial_nodes=64,angle_nodes=512,damping=0):
    """Deterministic polar transform of exact kitten CF, for checking convention and quadrature."""
    theta=(np.arange(angle_nodes)+.5)*np.pi/angle_nodes
    nquad=radial_nodes
    nodes,weights=leggauss(nquad)
    tmax=cut/np.maximum(abs(np.cos(theta)),abs(np.sin(theta)))
    t=.5*(1+nodes[None,:])*tmax[:,None]
    wt=.5*tmax[:,None]*weights[None,:]
    # phi_model=exp(-t^2/4)*(1-A*t^2/2 + i B*t*cos(theta))
    p=.6810463;amp=.3; refl2=.925
    c0=np.sqrt(1-refl2)/np.sqrt(1-refl2+amp*amp); c1=amp/np.sqrt(1-refl2+amp*amp)
    aa=p*c1*c1; bb=np.sqrt(2)*p*c0*c1
    # radial signed includes conjugate, for theta in [0,pi) and t>0: Re[phi * kernel]
    # phase factor exp(i sqrt2*t*alpha*cos theta)
    z=np.sqrt(2)*t[:,:,None]*alpha[None,None,:]*np.cos(theta)[:,None,None]
    real=(1-aa*t*t/2)[:,:,None]*np.cos(z)-(bb*t*np.cos(theta)[:,None])[:,:,None]*np.sin(z)
    outputs=[]
    for r in rs:
        pref=t*wt*np.exp(-.5*t*t-.25*damping*damping*t*t)*eval_laguerre(r,t*t/2)
        outputs.append((pref[:,:,None]*real).sum(axis=(0,1))/angle_nodes)
    return np.stack(outputs)




# ======================================================================
# FINITE-GRID PHYSICALITY DIAGNOSTICS
# Extracted from kqse_physicality_audit.py; all helpers self-contained below.
# ======================================================================
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_uniform_symmetric_grid(mu: np.ndarray, nu: np.ndarray) -> tuple[float, float]:
    if mu.ndim != 1 or nu.ndim != 1 or len(mu) % 2 != 1 or len(nu) % 2 != 1:
        raise ValueError("Odd, one-dimensional CF grid coordinates are required")
    dmu = float(mu[1] - mu[0]); dnu = float(nu[1] - nu[0])
    if dmu <= 0 or dnu <= 0:
        raise ValueError("CF grid coordinates must be increasing")
    if not np.allclose(np.diff(mu), dmu, atol=1e-12, rtol=0):
        raise ValueError("Nonuniform mu grid")
    if not np.allclose(np.diff(nu), dnu, atol=1e-12, rtol=0):
        raise ValueError("Nonuniform nu grid")
    if not np.allclose(mu, -mu[::-1], atol=1e-12, rtol=0):
        raise ValueError("mu grid is not centered")
    if not np.allclose(nu, -nu[::-1], atol=1e-12, rtol=0):
        raise ValueError("nu grid is not centered")
    return dmu, dnu


def photon_tomograms_all_r(phi: np.ndarray, mu: np.ndarray,
                          nu: np.ndarray, alpha: np.ndarray,
                          rmax: int) -> np.ndarray:
    """Eq. (wr_kernel_main) evaluated on the real-displacement slice.

    K_r(alpha; mu,nu)=exp[-(mu^2+nu^2)/4] L_r((mu^2+nu^2)/2)
                      * exp(i sqrt(2) mu alpha), alpha real.
    """
    dmu, dnu = validate_uniform_symmetric_grid(mu, nu)
    MU, NU = np.meshgrid(mu, nu, indexing="ij")
    rho2 = MU**2 + NU**2
    phase = np.exp(1j * np.sqrt(2.0) * np.outer(alpha, mu))
    damped_phi = phi * np.exp(-rho2 / 4.0)
    out = np.empty((rmax+1, len(alpha)), dtype=np.complex128)
    for r in range(rmax+1):
        weighted_mu = np.sum(damped_phi * eval_laguerre(r, rho2 / 2.0), axis=1)
        out[r] = (dmu * dnu / TWO_PI) * (phase @ weighted_mu)
    return out


def wigner_on_square(phi: np.ndarray, mu: np.ndarray,
                     nu: np.ndarray, extent: float, ngrid: int) -> tuple[float, float]:
    """Finite-window integral of W, not a full-plane normalization test."""
    dmu, dnu = validate_uniform_symmetric_grid(mu, nu)
    q = np.linspace(-extent, extent, ngrid)
    p = q.copy()
    left = np.exp(-1j * np.outer(q, mu))
    right = np.exp(-1j * np.outer(nu, p))
    W = (dmu*dnu / TWO_PI**2) * (left @ phi @ right)
    W = np.asarray(W)
    dq = q[1]-q[0]; dp = p[1]-p[0]
    integral = float((dq*dp*W.sum()).real)
    return integral, float(np.max(np.abs(W.imag)))


def quantum_bochners(phi: np.ndarray, mu: np.ndarray,
                     nu: np.ndarray, steps: tuple[int, ...]) -> list[dict]:
    """Finite Weyl-characteristic PSD tests with [q,p]=i.

    M[j,k] = <U(xi_j)^dagger U(xi_k)>
           = exp[i sigma(xi_j,xi_k)/2] phi(xi_k-xi_j).
    sigma((mu,nu),(mu',nu')) = mu nu' - nu mu'.
    Differences are exact CF-grid nodes; no interpolation is used.
    A negative eigenvalue disproves quantum positivity for this CF;
    nonnegative eigenvalues are necessary but not sufficient.
    """
    dmu, dnu = validate_uniform_symmetric_grid(mu, nu)
    ci, cj = len(mu)//2, len(nu)//2
    rows = []
    for s in steps:
        p = [(0,0),(s,0),(-s,0),(0,s),(0,-s),
             (s,s),(-s,-s),(s,-s),(-s,s)]
        if max(abs(pj[0]-pk[0]) for pj in p for pk in p) > ci:
            continue
        if max(abs(pj[1]-pk[1]) for pj in p for pk in p) > cj:
            continue
        M = np.empty((len(p),len(p)), dtype=np.complex128)
        for j,(aj,bj) in enumerate(p):
            for k,(ak,bk) in enumerate(p):
                sigma = dmu*dnu * (aj*bk - bj*ak)
                M[j,k] = np.exp(0.5j*sigma) * phi[ci + (ak-aj), cj + (bk-bj)]
        nonhermitian = float(np.max(np.abs(M-M.conj().T)))
        eigenvalues = np.linalg.eigvalsh((M+M.conj().T)/2)
        rows.append(dict(step=s, n_points=len(p),
                         min_eigenvalue=float(eigenvalues[0]),
                         negative_eigenvalues=int(np.count_nonzero(eigenvalues < -1e-10)),
                         hermiticity_residual=nonhermitian))
    return rows


def ensure_saved_r03_match(arrays: np.lib.npyio.NpzFile,
                           variant: str, computed: np.ndarray) -> float:
    if variant not in ('nocorr', 'corr'):
        return 0.0
    max_error = 0.0
    for r in range(4):
        name = f'w_kqse_{variant}_r{r}'
        if name not in arrays.files:
            raise ValueError(f"Missing saved validation array: {name}")
        saved = np.asarray(arrays[name], dtype=np.complex128)
        error = float(np.max(np.abs(computed[r]-saved)))
        max_error = max(max_error, error)
    if max_error > 1e-9:
        raise ValueError(f"Reconstruction disagrees with saved r=0..3 arrays: {max_error:g}")
    return max_error


def tex_table(rows: list[dict], path: Path) -> None:
    # Experimental variants only; journal-ready fragment.
    data = {(a['variant'], a['r']): a for a in rows}
    def fmt(x): return f"{x:.5f}"
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Finite-grid nonnegativity diagnostics for the experimental KQSE photon-number tomograms on the real displacement slice $\alpha\in[-5,5]$ (121 points). The negative-point count $N_-$ is calculated separately for each $r$.}',
        r'\label{tab:kqse_physicality_diagnostics}',
        r'\begin{tabular}{c rr rr}',
        r'\toprule',
        r'& \multicolumn{2}{c}{Without noise correction} & \multicolumn{2}{c}{With noise correction ($\kappa=0.9$)} \\',
        r'$r$ & $\min \operatorname{Re}\widehat{\mathsf w}_r$ & $N_-$ & $\min \operatorname{Re}\widehat{\mathsf w}_r$ & $N_-$ \\',
        r'\midrule',
    ]
    for r in range(4):
        a,b = data[('nocorr',r)], data[('corr',r)]
        lines.append(f"{r} & {fmt(a['min_value'])} & {a['negative_grid_points']} & {fmt(b['min_value'])} & {b['negative_grid_points']} " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '']
    path.write_text('\n'.join(lines),encoding='utf-8')


def analyze(input_path: Path, output_dir: Path, rmax: int = 60, plots: bool = False) -> None:
    """Compute the diagnostics using a saved experimental KQSE CF grid."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if rmax < 3:
        raise ValueError("rmax must be at least 3")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_path,allow_pickle=False) as source:
        mu=np.asarray(source['exp_mu_grid'], dtype=float)
        nu=np.asarray(source['exp_nu_grid'], dtype=float)
        alpha=np.asarray(source['exp_alpha_eval'], dtype=float)
        dmu,dnu=validate_uniform_symmetric_grid(mu,nu)
        if alpha.ndim != 1 or not np.allclose(alpha.imag, 0):
            raise ValueError('Expected 1D real displacement grid')
        phi_dict={variant: np.asarray(source[key], dtype=complex)
                  for variant,key,_ in VARIANTS}
        all_w={}
        validation={}
        for variant,phi in phi_dict.items():
            if phi.shape!=(len(mu),len(nu)):
                raise ValueError(f'CF grid shape mismatch: {variant}')
            all_w[variant]=photon_tomograms_all_r(phi,mu,nu,alpha,rmax)
            validation[variant]=ensure_saved_r03_match(source,variant,all_w[variant])

    np.savez_compressed(
        output_dir / 'photon_tomograms_r0_to_rmax.npz',
        alpha=alpha,
        r_values=np.arange(rmax + 1),
        w_model=all_w['model'],
        w_nocorr=all_w['nocorr'],
        w_corr=all_w['corr'],
    )

    photon_rows=[]; cf_rows=[]; bochner_rows=[]; cutoff_rows=[]; wigner_rows=[]
    origin=(len(mu)//2,len(nu)//2)
    tested_steps=(1,2,3,4,6,8,12)
    cutoffs=sorted(set([3,rmax]+[x for x in [10,20,40] if x<=rmax]))
    for variant,source_name,display_name in VARIANTS:
        phi=phi_dict[variant]; w=all_w[variant]
        absphi=np.abs(phi)
        cf_rows.append(dict(variant=variant, description=display_name,
            phi_origin_real=float(phi[origin].real),
            phi_origin_imag=float(phi[origin].imag),
            origin_deviation=float(abs(phi[origin]-1)),
            max_hermitian_symmetry_error=float(np.max(np.abs(phi-np.conj(phi[::-1,::-1])))),
            max_cf_modulus=float(absphi.max()),
            maximum_cf_modulus_excess=float(np.maximum(absphi-1,0).max()),
            n_cf_grid_points_over_unit_modulus=int(np.count_nonzero(absphi>1+1e-10)),
            reconstruction_validation_max_error=validation[variant]))

        for row in quantum_bochners(phi,mu,nu,tested_steps):
            bochner_rows.append(dict(variant=variant,**row))
        for r in range(4):
            v=w[r]; real=v.real; neg=np.maximum(-real,0)
            photon_rows.append(dict(variant=variant,r=r,
                min_value=float(real.min()),
                alpha_at_min=float(alpha[np.argmin(real)]),
                maximum_value=float(real.max()),
                negative_grid_points=int(np.count_nonzero(real<0)),
                max_negative_deviation=float(neg.max()),
                mean_squared_negative_part=float(np.mean(neg**2)),
                max_imaginary_part=float(np.max(np.abs(v.imag))),
                n_grid_points_over_one=int(np.count_nonzero(real>1)),
                max_excess_over_one=float(np.maximum(real-1,0).max())))

        for cutoff in cutoffs:
            p=w[:cutoff+1].real
            S=p.sum(axis=0)
            Nminus=np.maximum(-p,0).sum(axis=0)
            # This is a truncated sum. A deficit below 1 is not evidence
            # of nonnormalization without controlling the omitted tail.
            k0=int(np.argmin(np.abs(alpha)))
            idxmaxmass=int(np.argmax(Nminus))
            cutoff_rows.append(dict(variant=variant,rmax=cutoff,
                min_partial_sum=float(S.min()),max_partial_sum=float(S.max()),
                max_partial_sum_excess_above_one=float(np.maximum(S-1,0).max()),
                max_abs_partial_normalization_residual=float(np.max(np.abs(S-1))),
                partial_sum_at_alpha_zero=float(S[k0]),
                abs_residual_at_alpha_zero=float(abs(S[k0]-1)),
                max_truncated_negative_mass=float(Nminus.max()),
                alpha_at_max_truncated_negative_mass=float(alpha[idxmaxmass]),
                truncated_negative_mass_at_alpha_zero=float(Nminus[k0]),
                min_reference_comment='Finite cutoff only; not full normalization'))
        for extent,ngrid in [(4.,81),(6.,121)]:
            integral,im=wigner_on_square(phi,mu,nu,extent,ngrid)
            wigner_rows.append(dict(variant=variant, q_p_square_extent=extent,
                                    evaluation_grid_n=ngrid,
                                    finite_window_integral=integral,
                                    residual_from_one=integral-1.,max_imaginary_part=im))

    write_csv(output_dir/'cf_constraints.csv',cf_rows)
    write_csv(output_dir/'photon_r0_r3.csv',photon_rows)
    write_csv(output_dir/'quantum_bochners.csv',bochner_rows)
    write_csv(output_dir/'fock_cutoff_diagnostics.csv',cutoff_rows)
    write_csv(output_dir/'wigner_finite_window.csv',wigner_rows)
    tex_table(photon_rows, output_dir/'photon_diagnostics_table.tex')

    report=[
       '# Physicality diagnostics for experimental KQSE',
       '',
       f'Input: `{input_path.name}`; CF grid {len(mu)} x {len(nu)}; '
       f'displacement grid {len(alpha)} real points from {alpha[0]:g} to {alpha[-1]:g}.',
       f'Photon numbers reconstructed: 0,...,{rmax}. '
       'All r=0,...,3 values were checked against the saved arrays.',
       '',
       '## Scope',
       '',
       '- Negative photon probabilities are genuine violations of the necessary '
       'nonnegativity condition, as evaluated on the stated grid.',
       '- The finite-r negative mass `sum_{r=0}^R max(-Re(w_r),0)` is a *truncated* '
       'diagnostic. It is not the total negative probability mass.',
       '- `sum_{r=0}^R Re(w_r)` is a *partial sum*, not a directly observable '
       'full normalization error. A partial sum greater than 1 is incompatible '
       'with a physical distribution; a partial sum below 1 is inconclusive.',
       '- `phi(0,0)=1` and `phi(-mu,-nu)=conj(phi(mu,nu))` were explicitly '
       'enforced when the KQSE CF grids were reconstructed. Their agreement '
       'is therefore not an independent validation.',
       '- Wigner-function negativity is physically allowed. '
       'The Wigner integral is computed over a finite square and is not '
       'a full-plane normalization certificate.',
       '- The quantum-Bochner test uses exact nodes on the saved CF grid '
       'with [q,p]=i. A significantly negative eigenvalue violates a '
       'necessary quantum positivity condition. Passing finitely many '
       'tests does not guarantee physicality.',
       '- The reference model is included to expose numerical integration '
       'and finite-cutoff artifacts.',
       '- These data-driven reconstructions use plug-in bandwidths and '
       'a phenomenological noise correction; the oracle convergence '
       'theorem does not automatically cover them.',
       '',
       '## r = 0,...,3 (121 real-displacement points)',
       '',
       '| Variant | r | min Re(w_r) | negative grid points |',
       '|:---|---:|---:|---:|',
    ]
    for row in photon_rows:
        if row['variant']=='model': continue
        report.append(f"| {row['variant']} | {row['r']} | {row['min_value']:.6f} | {row['negative_grid_points']} |")
    report += ['', '## Quantum-Bochner necessary condition', '',
               'Nine-point Weyl test sets at the integer grid step indicated below.', '',
               '| Variant | step | minimum eigenvalue |',
               '|:---|---:|---:|']
    for row in bochner_rows:
        if row['step'] in [1,2,4,6]:
            report.append(f"| {row['variant']} | {row['step']} | {row['min_eigenvalue']:.6g} |")
    report += ['', '## Finite photon cutoff diagnostics', '',
               '| Variant | R | max sum_{r<=R} w_r | max sum_{r<=R} (-w_r)_+ |',
               '|:---|---:|---:|---:|']
    for row in cutoff_rows:
        report.append(f"| {row['variant']} | {row['rmax']} | {row['max_partial_sum']:.6f} | {row['max_truncated_negative_mass']:.6f} |")
    report += ['', '## Wigner finite-window integral', '',
               '| Variant | square | integral |', '|:---|:---|---:|']
    for row in wigner_rows:
        report.append(f"| {row['variant']} | [-{row['q_p_square_extent']:g},{row['q_p_square_extent']:g}]^2 | {row['finite_window_integral']:.7f} |")
    (output_dir/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

    if plots:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.5))
        for variant,_,name in VARIANTS:
            if variant=='model': continue
            Nminus=np.maximum(-all_w[variant][:4].real,0).sum(axis=0)
            ax.plot(alpha,Nminus,label=name)
        ax.set(xlabel=r'Real displacement $\alpha$',
               ylabel=r'Truncated negative mass $\sum_{r=0}^{3}(-\widehat{w}_r)_+$')
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(output_dir/'negative_mass_r0_r3.png',dpi=190)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for variant,_,name in VARIANTS:
            S=all_w[variant][:rmax+1].real.sum(axis=0)
            ax.plot(alpha,S,label=name)
        ax.axhline(1,linestyle='--',color='black',linewidth=1)
        ax.set(xlabel=r'Real displacement $\alpha$',
               ylabel=fr'Partial sum $\sum_{{r=0}}^{{{rmax}}}\mathrm{{Re}}\widehat{{w}}_r$')
        ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(output_dir/'finite_cutoff_probability_sum.png',dpi=190)
        plt.close(fig)

    print(f'Results written to {output_dir.resolve()}')
    for row in photon_rows:
        if row['variant'] != 'model':
            print(f"  {row['variant']:6s} r={row['r']} "
                  f"min={row['min_value']:+.6f} negative={row['negative_grid_points']}")
    for row in bochner_rows:
        if row['step']==2:
            print(f"  Weyl PSD: {row['variant']:6s} step=2 "
                  f"lambda_min={row['min_eigenvalue']:+.6g}")



# ======================================================================
# KITTEN MONTE CARLO / TRUNCATED PILOT
# Extracted from kqse_kitten_montecarlo.py; all helpers self-contained below.
# ======================================================================
def distribution_cdf(x, theta):
    # from rho=(1-p)|0><0|+p|c0 0+c1 1><...|, q vacuum variance 1/2.
    # f(x|theta)= exp(-x*x)/sqrt(pi) * (1-A+2*A*x*x+2*B*cos(theta)*x).
    pdf0=np.exp(-x*x)/sqrt_pi
    return .5*(1+erf(x)) - pdf0*(A*x+B*np.cos(theta))


def sample_centers(counts, rng, cdf_grids, x_grid):
    return [np.interp(rng.random(n), cdf, x_grid) for n,cdf in zip(counts,cdf_grids)]


def construct_geometry(mu,nu,N_theta=20):
    MU,NU=np.meshgrid(mu,nu,indexing='ij')
    radius=np.hypot(MU,NU)
    theta=np.mod(np.arctan2(NU,MU),2*np.pi)
    sign=np.where(theta>=np.pi,-1,+1)
    folded=np.where(theta>=np.pi,theta-np.pi,theta)
    bins=np.clip(np.floor(folded / (np.pi/N_theta)).astype(int),0,N_theta-1)
    return radius,bins,sign


def gaussian_pilot_phi_truncated(samples, t_values):
    """Exact FT of Silverman gaussian_kde on [-3,3], normalized to unit mass.
    Matches np.trapezoid(pdf*exp(itx)) of gaussian_kde after finite-window renorm.
    """
    n=len(samples)
    sigma=np.std(samples,ddof=1)* (3*n/4)**(-1/5)
    if sigma<=0: raise ValueError('Degenerate sample!')
    lb,ub=-3.,3.
    Z=np.mean(ndtr((ub-samples)/sigma)-ndtr((lb-samples)/sigma))
    out=np.empty(len(t_values),dtype=np.complex128)
    for sl in range(0,len(t_values),128):
        ts=t_values[sl:sl+128,None]
        xs=samples[None,:]
        factor=np.exp(1j*ts*xs-.5*sigma*sigma*ts*ts)
        lo=ndtr((lb-xs-1j*sigma*sigma*ts)/sigma)
        hi=ndtr((ub-xs-1j*sigma*sigma*ts)/sigma)
        out[sl:sl+128]=np.mean(factor*(hi-lo),axis=1)/Z
    return out


def cf_from_group(samples,mu,nu,radius,bins,sign,noise_correction=False,kappa=.9):
    ntheta=len(samples); flat_bins=bins.ravel(); flat_radius=radius.ravel();flat_sign=sign.ravel()
    phi=np.ones(flat_radius.shape,dtype=np.complex128)
    bw=np.zeros(flat_radius.shape,dtype=float)
    for b in range(ntheta):
        mask=(flat_bins==b) & (flat_radius>1e-12)
        rr=flat_radius[mask]
        if not len(rr):continue
        # near identical radii get exactly the same bandwidth as original, no rounding needed
        t=rr/(kappa if noise_correction else 1.)
        x=samples[b]
        # original kde is evaluated separately for sample_sign*(-1), but [-3,3]
        # symmetry and conjugation give the same result
        absphi=np.abs(gaussian_pilot_phi_truncated(x,t))
        absphi=np.maximum(absphi,np.finfo(float).tiny)
        rawbw=2./t*np.sqrt(np.maximum(0.,1.-absphi**2)/(len(x)*absphi**2))
        bb=np.clip(rawbw,.01,.5)
        empirical=np.exp(1j*t[:,None]*x[None,:]).mean(axis=1)
        vals=empirical*np.exp(-.25*(bb*t)**2)
        if noise_correction:
            vals=vals*np.exp(.5*(((1-kappa)/kappa)*rr)**2)
        vals=np.where(flat_sign[mask]==1,vals,np.conjugate(vals))
        phi[mask]=vals
        bw[mask]=bb
    phi=phi.reshape(radius.shape)
    # exactly as in actual Colab
    phi=.5*(phi+np.conjugate(phi[::-1,::-1]))
    i0=np.argmin(np.abs(mu));j0=np.argmin(np.abs(nu));phi[i0,j0]=1.+0j
    return phi,bw.reshape(radius.shape)


def kernel_transform_matrix(mu,nu,alpha,rlist=(0,1,2,3)):
    MU,NU=np.meshgrid(mu,nu,indexing='ij')
    dmu=mu[1]-mu[0];dnu=nu[1]-nu[0]
    x=(MU*MU+NU*NU).ravel()/2
    base=np.exp(-x/2)
    phase=np.exp(1j*np.sqrt(2)*np.outer(alpha,MU.ravel()))
    return np.concatenate([(dmu*dnu/(2*np.pi))*(phase*(base*eval_laguerre(r,x))[None,:]) for r in rlist],axis=0)


def measure(phi,kmat,n_alpha,n_r=4):
    w=(kmat@phi.ravel()).reshape((n_r,n_alpha))
    if np.max(abs(w.imag))>5e-9:raise AssertionError('nonreal reconstructed tomogram')
    w=w.real
    neg=np.maximum(-w,0)
    mins=w.min(axis=1)
    score={f'min_r{r}':mins[r] for r in range(n_r)}
    score.update({f'neg_nodes_r{r}':int((w[r]<0).sum()) for r in range(n_r)})
    score['min_all_r0_r3']=float(w.min())
    score['max_negative_mass_r0_r3']=float(neg.sum(axis=0).max())
    pos=np.maximum(w,0).sum(axis=0)
    score['max_negative_ratio_r0_r3']=float(np.max(np.divide(neg.sum(axis=0),pos,out=np.zeros_like(pos),where=pos>0)))
    score['mean_negative_mass_r0_r3']=float(neg.sum(axis=0).mean())
    score['max_partial_sum_r0_r3']=float(w.sum(axis=0).max())
    score['frac_alpha_any_negative']=float(np.mean(neg.sum(axis=0)>1.e-12))
    return score




# ======================================================================
# SIX-STATE SYNTHETIC TOMOGRAPHY
# Extracted from kqse_simple_states_physicality.py; all helpers self-contained below.
# ======================================================================
def analytic_cf(name,MU,NU):
    u=MU*MU+NU*NU
    if name=='vacuum': return np.exp(-u/4).astype(complex)
    if name=='coherent': return np.exp(-u/4+1j*(1.3*MU+.4*NU))
    if name=='thermal': return np.exp(-(2*.8+1)*u/4).astype(complex)
    if name=='fock1': return (np.exp(-u/4)*(1-u/2)).astype(complex)
    if name=='squeezed': return np.exp(-.25*(np.exp(-1.1)*MU**2+np.exp(1.1)*NU**2)).astype(complex)
    if name=='kitten': return np.exp(-u/4)*(1-A*u/2+1j*B*MU)
    raise ValueError(name)


def kitten_inverse_cdf(u,theta):
    q=np.clip(erfinv(2*u-1),-5,5)
    for it in range(14):
        g=np.exp(-q*q)/np.sqrt(np.pi)
        F=.5*(1+erf(q))-g*(A*q+B*np.cos(theta))
        f=g*(1-A+2*A*q*q+2*B*q*np.cos(theta))
        q-=np.clip((F-u)/np.maximum(f,1e-12),-.55,.55)
    return q


def photon_shots(name,theta,rng):
    n=len(theta)
    if name=='vacuum': return rng.normal(0,2**-.5,n)
    if name=='coherent':return (1.3*np.cos(theta)+.4*np.sin(theta))+rng.normal(0,2**-.5,n)
    if name=='thermal':return rng.normal(0,np.sqrt(1.3),n)
    if name=='squeezed':return rng.standard_normal(n)*np.sqrt(.5*(np.exp(-1.1)*np.cos(theta)**2+np.exp(1.1)*np.sin(theta)**2))
    u=rng.uniform(1e-10,1-1e-10,n)
    if name=='kitten':return kitten_inverse_cdf(u,theta)
    if name=='fock1':
        q=np.clip(erfinv(2*u-1),-5,5)
        for _ in range(14):
            g=np.exp(-q*q)/np.sqrt(np.pi)
            F=.5*(1+erf(q))-q*g
            f=2*q*q*g
            # near-zero derivative: bisection used below for robust inverse
            step=np.clip((F-u)/np.maximum(f,1e-12),-.2,.2)
            q-=step
        # Robust fallback if Newton is slow near zero: use monotone inverse table
        # (in this family f is independent of angle)
        xgrid=np.linspace(-6,6,40001)
        cdf=.5*(1+erf(xgrid))-xgrid*np.exp(-xgrid*xgrid)/np.sqrt(np.pi)
        return np.interp(u,cdf,xgrid)
    raise ValueError(name)


def geometry(mu,nu,bins):
    rr,th,sgn=cart_grid_geometry(mu,nu)
    idx=np.minimum((th/(np.pi/bins)).astype(int),bins-1)
    return rr.ravel(),idx.ravel(),sgn.ravel()


def estimate_emp_and_kcfe(theta,x,mu,nu,nbins,geom):
    # Generate directly on [0,pi); no extra experiment-specific sign convention.
    ids=np.minimum((theta/(np.pi/nbins)).astype(int),nbins-1)
    rr,bi,sgn=geom
    empirical=np.ones(len(rr),dtype=np.complex128)
    regular=np.ones(len(rr),dtype=np.complex128)
    h_grid=np.full(len(rr),np.nan)
    for k in range(nbins):
        indices=np.where((bi==k)&(rr>1e-9))[0]
        if not indices.size:continue
        sample=x[ids==k]
        n=len(sample)
        if n<8:raise ValueError(f'Insufficient shots in bin {k}: {n}')
        tt=rr[indices]
        # Average unique t values; speeds high-frequency eval on Cartesian grid.
        ut,inv=np.unique(np.round(tt,12),return_inverse=True)
        raw_u=np.exp(1j*ut[:,None]*sample[None,:]).mean(axis=1)
        raw=raw_u[inv]
        # Original fully data-driven Gaussian KCFE, with full-line Silverman pilot.
        bw=np.std(sample,ddof=1)*(3*n/4)**(-1/5)
        pilot=np.clip(np.abs(raw_u)*np.exp(-.5*(bw*ut)**2),1.e-140,1)
        h=np.clip((2/ut)*np.sqrt(np.maximum(0,1-pilot*pilot)/(n*pilot*pilot)),.01,.5)[inv]
        vv=raw*np.exp(-.25*(tt*h)**2)
        empirical[indices]=np.where(sgn[indices]>0,raw,raw.conj())
        regular[indices]=np.where(sgn[indices]>0,vv,vv.conj())
        h_grid[indices]=h
    return sanitize_cf(empirical.reshape(len(mu),len(nu)),mu,nu),sanitize_cf(regular.reshape(len(mu),len(nu)),mu,nu),h_grid


def oracle_center_cf(name,mu,nu,nbins,geom):
    rr,bi,sgn=geom
    central=(bi+.5)*np.pi/nbins
    xx=rr*np.cos(central);yy=rr*np.sin(central)
    out=analytic_cf(name,xx,yy)
    out=np.where(sgn>0,out,out.conj()).reshape(len(mu),len(nu))
    return sanitize_cf(out,mu,nu)


def metrics(phi,K,alpha,ref,mu,nu):
    w=photon_from_phi(phi,K,n_alpha=len(alpha),nr=4)
    w_clip=np.maximum(w,0)
    negative=np.maximum(0,-w);positive=np.maximum(w,0)
    mass=negative.sum(axis=0);posmass=positive.sum(axis=0)
    at=int(mass.argmax())
    return dict(min_r0=float(w[0].min()),min_r1=float(w[1].min()),min_r2=float(w[2].min()),min_r3=float(w[3].min()),
        negmass_max=float(mass[at]),negative_to_positive_at_massmax=float(mass[at]/posmass[at]) if posmass[at]>0 else np.nan,
        negmass_integrated=float(np.trapezoid(mass,x=alpha)),
        ise_total=float(np.trapezoid(np.sum((w-ref)**2,axis=0),x=alpha)),
        ise_r2=float(np.trapezoid((w[2]-ref[2])**2,x=alpha)),
        ise_clip_total=float(np.trapezoid(np.sum((w_clip-ref)**2,axis=0),x=alpha)),
        clip_sum_max=float(w_clip.sum(axis=0).max()),
        bochner_min=float(bochner_eigen(phi,mu,nu)),
        negnodes=int((w<-1e-6).sum()),
        normpartial_max=float(w.sum(axis=0).max()))


def simulate_six_states(outdir,shots,trials,seed,grid_n=41,bins=(20,40)):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    alpha=np.linspace(-3,3,73)
    # Maintain SAME truncation window as manuscript (halfwidth 6.5 approximately).
    mu=np.linspace(-6.5,6.5,grid_n);nu=mu.copy(); MU,NU=np.meshgrid(mu,nu,indexing='ij')
    K=photon_kernel_mat(mu,nu,alpha,rs=(0,1,2,3))
    radius2=MU**2+NU**2
    rng=np.random.default_rng(seed)
    records=[]; baseline=[];start=time.time()
    for state in STATES:
        truephi=analytic_cf(state,MU,NU)
        truew=photon_from_phi(truephi,K,n_alpha=len(alpha))
        for bin_n in bins:
            geom=geometry(mu,nu,bin_n)
            for label,phi in [('exact CF',truephi),('exact CF with angle snapping',oracle_center_cf(state,mu,nu,bin_n,geom))]:
                base=metrics(phi,K,alpha,truew,mu,nu)
                baseline.append(dict(state=state,bins=bin_n,method=label,**base))
            for rep in range(trials):
                theta=rng.uniform(0,np.pi,shots)
                x=photon_shots(state,theta,rng)
                emp,kcfe,h=estimate_emp_and_kcfe(theta,x,mu,nu,bin_n,geom)
                for method,phi in [('empirical',emp),('KCFE plug-in',kcfe),
                                   ('empirical + fixed Gaussian 0.35',emp*np.exp(-(.35**2)*radius2/4)),
                                   ('empirical + fixed Gaussian 0.65',emp*np.exp(-(.65**2)*radius2/4)),
                                   ('KCFE + fixed Gaussian 0.35',kcfe*np.exp(-(.35**2)*radius2/4))]:
                    rec=metrics(phi,K,alpha,truew,mu,nu)
                    records.append(dict(state=state,bins=bin_n,trial=rep,shots=shots,method=method,**rec))
                if (rep+1)%10==0:print(f'{state}, {bin_n} bins, run {rep+1}/{trials}, {time.time()-start:.1f}s',flush=True)
        print('FINISHED',state,'elapsed',round(time.time()-start,1),flush=True)
    import pandas as pd
    d=pd.DataFrame(records);b=pd.DataFrame(baseline)
    d.to_csv(outdir/'trials.csv',index=False);b.to_csv(outdir/'exact_controls.csv',index=False)
    g=d.groupby(['state','bins','method']).agg(trials=('trial','size'),mean_ise=('ise_total','mean'),
      median_ise=('ise_total','median'),mean_ise_clip=('ise_clip_total','mean'),mean_negmass_max=('negmass_max','mean'),median_negmass_max=('negmass_max','median'),
      q90_negmass_max=('negmass_max',lambda s:s.quantile(.90)),
      median_bochnermin=('bochner_min','median'),
      median_min_r2=('min_r2','median'),median_negnodes=('negnodes','median'),
      frac_any_negative=('negnodes',lambda s:(s>0).mean()),
      frac_bochner_negative=('bochner_min',lambda s:(s< -1e-6).mean())).reset_index()
    g.to_csv(outdir/'summary.csv',index=False)
    print('WROTE',outdir,'seconds',round(time.time()-start,1),flush=True)
    for s in STATES:
        print('STATE',s)
        print(g[(g.state==s)&(g.bins==20)][['method','mean_ise','median_negmass_max','median_min_r2','median_bochnermin','frac_any_negative']].to_string(index=False,float_format=lambda t:f'{t:.5g}'),flush=True)
    return d,b,g



# ======================================================================
# END-TO-END EXPERIMENTAL RECONSTRUCTION FROM RAW PAIRS
# ======================================================================
def reconstruct_experiment(data_path, output, checkpoint=None):
    theta, x = np.loadtxt(data_path, unpack=True)
    # Grid from the published analysis: endpoint-excluded symmetric discretization.
    mu = (np.arange(61)-30) * (13.0/61)
    nu = mu.copy()
    alpha = np.linspace(-5,5,121)
    MU, NU = np.meshgrid(mu,nu,indexing='ij')
    phi_model = analytic_cf('kitten', MU, NU)
    radius,bi,sign = construct_geometry(mu,nu,20)
    theta_fold,xf = original_phase_convention(theta,x)
    ids = np.clip(np.floor(theta_fold/(np.pi/20)).astype(int),0,19)
    samples = [xf[ids==j] for j in range(20)]
    if sum(len(y) for y in samples)!=len(theta):
        raise RuntimeError('Phase-bin assignment lost measurements')
    print('Observed phase-bin counts:',[len(y) for y in samples],flush=True)
    phi_no,bw_no=cf_from_group(samples,mu,nu,radius,bi,sign,False)
    phi_co,bw_co=cf_from_group(samples,mu,nu,radius,bi,sign,True)
    KM=kernel_transform_matrix(mu,nu,alpha)
    w_no=(KM@phi_no.ravel()).reshape(4,len(alpha)).real
    w_co=(KM@phi_co.ravel()).reshape(4,len(alpha)).real
    w_model=(KM@phi_model.ravel()).reshape(4,len(alpha)).real
    arrays={'exp_mu_grid':mu,'exp_nu_grid':nu,'exp_alpha_eval':alpha,
        'phi_model':phi_model,'phi_exp_kqse_nocorr':phi_no,
        'phi_exp_kqse_corr':phi_co,'bw_exp_nocorr':bw_no,
        'bw_exp_corr':bw_co}
    for r in range(4):
        arrays[f'w_kqse_nocorr_r{r}']=w_no[r]
        arrays[f'w_kqse_corr_r{r}']=w_co[r]
        arrays[f'w_model_r{r}']=w_model[r]
    if checkpoint is not None:
        with np.load(checkpoint,allow_pickle=False) as prev:
            diffs={key:float(np.max(np.abs(val-prev[key])))
                   for key,val in arrays.items() if key in prev.files}
        pd.Series(diffs,name='max_abs_difference').to_csv(output/'checkpoint_differences.csv')
        print('Checkpoint max CF difference:',max(diffs.get('phi_exp_kqse_corr',0),diffs.get('phi_exp_kqse_nocorr',0)),flush=True)
        if max(diffs.values())>2e-6:
            raise AssertionError('Raw reconstruction differs from supplied checkpoint')
    np.savez_compressed(output/'experimental_arrays.npz',**arrays)
    # Raw empirical characteristic function: same binned frequency assignment,
    # without either KCFE Gaussian attenuation or detector correction.
    er=radius.ravel(); ib=bi.ravel(); isign=sign.ravel()
    rawcf=np.ones_like(er,dtype=np.complex128)
    for j in range(20):
        idx=np.where((ib==j)&(er>1e-10))[0]
        raw=np.exp(1j*er[idx,None]*samples[j][None,:]).mean(axis=1)
        rawcf[idx]=np.where(isign[idx]>0,raw,raw.conj())
    rawcf=sanitize_cf(rawcf.reshape(len(mu),len(nu)),mu,nu)
    raww=photon_from_phi(rawcf,KM,n_alpha=len(alpha))
    np.savez_compressed(output/'unsmoothed_experiment.npz',phi_empirical=rawcf,w_empirical=raww)
    return arrays, samples, theta, x, raww


def test_angle_snapping(experimental,output):
    mu=experimental['exp_mu_grid'];nu=experimental['exp_nu_grid'];alpha=experimental['exp_alpha_eval']
    KM=photon_kernel_mat(mu,nu,alpha)
    rows=[]
    for n in (10,20,40,80,160):
        geom=geometry(mu,nu,n)
        cf=oracle_center_cf('kitten',mu,nu,n,geom)
        v,w=compute_metrics(str(n),'exact CF at nearest bin centers',cf,KM,experimental)
        rows.append(v)
    pd.DataFrame(rows).to_csv(output/'angle_snapping_exact.csv',index=False)
    return rows


def test_oracle_adaptive(shots,output):
    mu=np.linspace(-6.5,6.5,41);nu=mu.copy();alpha=np.linspace(-3,3,73)
    MU,NU=np.meshgrid(mu,nu,indexing='ij');rr=np.hypot(MU,NU)
    KM=photon_kernel_mat(mu,nu,alpha,rs=(0,1,2,3))
    rows=[]
    for state in STATES:
        exact=analytic_cf(state,MU,NU)
        wexact=photon_from_phi(exact,KM,len(alpha))
        for bins in (20,40):
            geom=geometry(mu,nu,bins)
            snapped=oracle_center_cf(state,mu,nu,bins,geom)
            for source,base in (('exact',exact),('angle snapped',snapped)):
                z=np.clip(np.abs(base),1e-140,1)
                h=(2/np.maximum(rr,1e-9))*np.sqrt(np.maximum(0,1-z*z)/(shots/bins*z*z))
                h=np.clip(h,.01,.5)
                for label,cf in (('none',base),
                    ('oracle adaptive h',base*np.exp(-.25*(rr*h)**2)),
                    ('fixed Gaussian 0.35',base*np.exp(-.25*.35**2*rr**2)),
                    ('fixed Gaussian 0.65',base*np.exp(-.25*.65**2*rr**2))):
                    result=metrics(cf,KM,alpha,wexact,mu,nu)
                    rows.append({'state':state,'bins':bins,'source':source,'filter':label,**result})
    d=pd.DataFrame(rows);d.to_csv(output/'synthetic_oracle_controls.csv',index=False)
    return d


def simulate_kitten_150(exp,samples,theta_data,output,n_trials=150,seed=261010):
    mu=exp['exp_mu_grid'];nu=exp['exp_nu_grid'];alpha=exp['exp_alpha_eval']
    radius,bi,sign=construct_geometry(mu,nu,20)
    kmat=kernel_transform_matrix(mu,nu,alpha)
    counts=np.array([len(a) for a in samples]);rng=np.random.default_rng(seed)
    xgrid=np.linspace(-5.5,5.5,24001)
    centers=(np.arange(20)+.5)*np.pi/20
    cdfs=[]
    for t in centers:
        q=distribution_cdf(xgrid,t)
        if np.min(np.diff(q))<-1e-12:raise RuntimeError('Invalid model CDF')
        cdfs.append(np.maximum.accumulate(np.clip(q,0,1)))
    # Previously used approximation for MC ONLY: full-line Gaussian pilot.
    # Analytic truncated-pilot integration is retained for the real data.
    def pilot_fast(x,t):
        n=len(x)
        sigma=np.std(x,ddof=1)*(3*n/4)**(-1/5)
        return np.exp(-.5*(sigma*t)**2) * np.exp(1j*t[:,None]*x[None,:]).mean(axis=1)
    global gaussian_pilot_phi_truncated
    original=gaussian_pilot_phi_truncated
    records=[]
    try:
        gaussian_pilot_phi_truncated=pilot_fast
        for scenario in ('ideal','assumed_gaussian_noise'):
            for trial in range(n_trials):
                draws=sample_centers(counts,rng,cdfs,xgrid)
                if scenario=='assumed_gaussian_noise':
                    draws=[.9*v+.1*rng.standard_normal(len(v)) for v in draws]
                for correction,kind in ((False,'nocorr'),(True,'corr')):
                    phi,_=cf_from_group(draws,mu,nu,radius,bi,sign,correction)
                    res=measure(phi,kmat,len(alpha))
                    records.append({'scenario':scenario,'trial':trial,'method':kind,**res})
            print(f'Kitten MC: {scenario}, {n_trials} realizations complete',flush=True)
    finally:
        gaussian_pilot_phi_truncated=original
    d=pd.DataFrame(records)
    d.to_csv(output/'kitten_montecarlo_trials.csv',index=False)
    obs={}
    for kind in ('nocorr','corr'):
        phi=exp['phi_exp_kqse_'+kind]
        obs[kind]=measure(phi,kmat,len(alpha))
    rows=[]
    for scenario in ('ideal','assumed_gaussian_noise'):
        for kind in ('nocorr','corr'):
            sub=d[(d.scenario==scenario)&(d.method==kind)]
            for field in ('min_r0','min_r1','min_r2','min_r3','max_negative_mass_r0_r3'):
                vals=sub[field].to_numpy();val=obs[kind][field]
                extreme=(vals<=val) if field.startswith('min_') else (vals>=val)
                rows.append({'scenario':scenario,'method':kind,'metric':field,
                    'observed':val,'sim_median':float(np.median(vals)),
                    'count_as_or_more_extreme':int(extreme.sum()),
                    'fraction_as_or_more_extreme':float(extreme.mean()),
                    'n_trials':len(vals)})
    s=pd.DataFrame(rows);s.to_csv(output/'kitten_montecarlo_summary.csv',index=False)
    return s


def phase_sensitivity(exp,theta,x,output):
    mu=exp['exp_mu_grid'];nu=exp['exp_nu_grid'];alpha=exp['exp_alpha_eval']
    KM=photon_kernel_mat(mu,nu,alpha)
    rows=[]
    for correction in (False,True):
        kind='corr' if correction else 'nocorr'
        for N in (20,40,80,160):
            phi=(exp['phi_exp_kqse_'+kind] if N==20
                 else cf_from_binned(theta,x,mu,nu,N,correction)[0])
            rec,_=compute_metrics(f'{N} bins',kind,phi,KM,exp)
            rows.append(rec)
        for width in (2.25,4.5,9.):
            phi,neff,h=cf_phase_local(theta,x,mu,nu,width,correction)
            rec,_=compute_metrics(f'local Gaussian {width:g} deg',kind,phi,KM,exp)
            rows.append(rec)
    pd.DataFrame(rows).to_csv(output/'experimental_phase_alternatives.csv',index=False)
    return rows


def make_audit_report(output,shot_count,exper,raw,angle,oracle,synth,mc,verbose=True):
    p=pd.read_csv(output/'experimental_audit'/'photon_r0_r3.csv')
    b=pd.read_csv(output/'experimental_audit'/'quantum_bochners.csv')
    c=pd.read_csv(output/'experimental_audit'/'fock_cutoff_diagnostics.csv')
    w=pd.read_csv(output/'experimental_audit'/'wigner_finite_window.csv')
    checks=[]
    def check(name,actual,expected,tol):
        checks.append(dict(name=name,actual=float(actual),expected=float(expected),
                           abs_difference=float(abs(actual-expected)),tolerance=tol,
                           passed=bool(abs(actual-expected)<=tol)))
    def pro(variant,r):
        return p[(p.variant==variant)&(p.r==r)].iloc[0]
    def mass(variant):
        return c[(c.variant==variant)&(c.rmax==60)].iloc[0]
    def bo(variant):
        return b[(b.variant==variant)&(b.step==2)].iloc[0]
    for variant,mins,neg in [
        ('nocorr',[-.002392,-.006444,-.044054,-.020604],[13,40,27,25]),
        ('corr',[-.002355,-.006947,-.004531,-.022076],[14,39,21,19])]:
        for r,(v,n) in enumerate(zip(mins,neg)):
            check(f'experimental {variant}: minimum r={r}',pro(variant,r).min_value,v,5e-6)
            check(f'experimental {variant}: negative nodes r={r}',pro(variant,r).negative_grid_points,n,0)
    for kind,N,ratio,S,eig in [('nocorr',.076485,.0711,1.007361,-.053459),
                               ('corr',.038018,.0366,1.006898,-.056216)]:
        m=mass(kind);check(kind+' max P60-',m.max_truncated_negative_mass,N,5e-6)
        a=exper['exp_alpha_eval'];mu=exper['exp_mu_grid'];nu=exper['exp_nu_grid'];
        res=photon_tomograms_all_r(exper['phi_exp_kqse_'+kind],mu,nu,a,60).real
        neg=np.maximum(-res,0).sum(axis=0);pos=np.maximum(res,0).sum(axis=0)
        idx=np.argmax(neg)
        check(kind+' P60-/P60+ (at max P60-)',neg[idx]/pos[idx],ratio,5e-5)
        check(kind+' max S60',m.max_partial_sum,S,5e-6)
        check(kind+' Bochner min',bo(kind).min_eigenvalue,eig,5e-6)
    check('reference max P60-',mass('model').max_truncated_negative_mass,.000129,2e-6)
    check('reference Bochner min',bo('model').min_eigenvalue,3.06e-6,1e-6)
    for kind,targ in [('nocorr',.998423),('corr',.996406)]:
        v=w[(w.variant==kind)&(w.q_p_square_extent==4)].iloc[0]
        check(kind+' Wigner integral [-4,4]^2',v.finite_window_integral,targ,5e-6)
    check('raw empirical min r2',raw[2].min(),-.09421,5e-5)
    oracle_angle=next(x for x in angle if x['setting']=='20')
    check('angle-snapped exact: Bochner min',oracle_angle['bochner_min'],-.04,5e-4)
    check('angle-snapped exact: r2 min',oracle_angle['min_r2'],-.000865,3e-6)
    so=oracle[(oracle.state=='fock1')&(oracle.bins==20)&(oracle.source=='exact')&(oracle['filter']=='oracle adaptive h')].iloc[0]
    check('fock1 oracle regularizer: maximum P3-',so.negmass_max,.0312,1e-4)
    for method,target in [('empirical',.0158),('KCFE plug-in',.0605)]:
        a=synth[(synth.state=='fock1')&(synth.bins==20)&(synth.method==method)].iloc[0]
        check('fock1 180 Monte Carlo: median P3- '+method,a.median_negmass_max,target,3e-4)
    for scenario,kind,field,k in [
        ('ideal','nocorr','min_r2',0),
        ('assumed_gaussian_noise','nocorr','min_r2',8),
        ('assumed_gaussian_noise','corr','min_r2',62),
        ('assumed_gaussian_noise','corr','min_r3',3)]:
        v=mc[(mc.scenario==scenario)&(mc.method==kind)&(mc.metric==field)].iloc[0]
        check(f'kitten {scenario}, {kind}, {field} count',v.count_as_or_more_extreme,k,0)
    d=pd.DataFrame(checks)
    d.to_csv(output/'manuscript_consistency.csv',index=False)
    failure=d[~d.passed]
    summary={'total_checks':len(d),'passed_checks':int(d.passed.sum()),
      'failed_checks':int((~d.passed).sum()),'failed':failure.name.tolist(),
      'shot_count':shot_count,'synthetic_state_trials_per_bin':int(synth.trials.max()),
      'kitten_sim_trials_per_scenario':int(mc.n_trials.max()),
      'warnings':['Pilot on real data uses analytically integrated truncated Gaussian KDE [-3,3].',
         'Kitten MC uses validated full-line Gaussian pilot approximation.',
         'Experimental kitten reference is approximate, not true unknown state.',
         'Oracle bandwidth from the exact CF is a deterministic mechanism test, not a practical estimator.',
         'The 41-point synthetic CF grid is distinct from the 61-point experimental grid.',
         'Finite Bochner tests only detect violations; never certify quantum physicality.',
         'No finite photon number cutoff establishes complete normalization.',
         'Phase smoothing and other alternative processing methods are exploratory.']}
    (output/'validation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    if verbose:
        print('\nMANUSCRIPT CONSISTENCY:',len(d)-len(failure),'/',len(d),'PASS',flush=True)
        if len(failure):print(failure.to_string(index=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Original DataPhase28.dat')
    parser.add_argument('--output',type=Path,default=Path('physicality_reproduction'))
    parser.add_argument('--checkpoint',type=Path,default=None,help='Optional prior reconstruction arrays for validation')
    parser.add_argument('--quick',action='store_true',help='Smoke test with 4 kitten and 5 six-state trials instead of 150/180; not manuscript validation')
    args=parser.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    print('1/6. Raw experimental reconstruction and KCFE',flush=True)
    exp,samples,theta,x,raww=reconstruct_experiment(args.data,out,args.checkpoint)
    print('2/6. Experimental physicality / extended photon reconstruction up to r=60',flush=True)
    analyze(out/'experimental_arrays.npz',out/'experimental_audit',rmax=60,plots=False)
    pd.DataFrame([{'r':r,'min_value':float(raww[r].min())} for r in range(4)]).to_csv(out/'unsmoothed_experimental_minima.csv',index=False)
    print('3/6. Exact-angle and synthetic oracle-bandwidth controls',flush=True)
    angle=test_angle_snapping(exp,out)
    oracle=test_oracle_adaptive(14153,out)
    print('4/6. Experimental phase-bin / local phase smoothing alternatives',flush=True)
    phase_sensitivity(exp,theta,x,out)
    trials=4 if args.quick else 150
    print(f'5/6. Kitten Monte Carlo: {trials} trials per scenario',flush=True)
    mc=simulate_kitten_150(exp,samples,theta,out,n_trials=trials)
    nsynth=5 if args.quick else 180
    print(f'6/6. Six synthetic states: {nsynth} trials per state/bins',flush=True)
    _,_,synth=simulate_six_states(out/'six_state_study',shots=14153,trials=nsynth,seed=2026101003,grid_n=41,bins=(20,40))
    result=make_audit_report(out,len(theta),exp,raww,angle,oracle,synth,mc)
    (out/'input_provenance.json').write_text(json.dumps({
        'data_file':args.data.name,'sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'shots':len(theta),'code_file':'kqse_physicality_all_in_one.py',
        'kitten_mc_seed':261010,'six_state_seed':2026101003,
        'quick':args.quick,'runtime_seconds':round(time.monotonic()-start,2),
        'numpy':np.__version__,'scipy':__import__('scipy').__version__,
        'pandas':pd.__version__},indent=2),encoding='utf8')
    print('Done in',round(time.monotonic()-start,2),'s; output=',out,flush=True)
    if not args.quick and result['failed_checks']:
        raise SystemExit('Manuscript mismatch: consult manuscript_consistency.csv')

if __name__=='__main__':
    main()
