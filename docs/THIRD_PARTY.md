# Scientific provenance of comparison methods

The following references describe third-party methods compared with KQSE in the article. Implementations of these methods were **not** included in this repository and were **not** retrained in the comparison notebooks: those notebooks load fixed, previously obtained reconstruction matrices.

- **BI:** J. C. Chapman et al., *Bayesian homodyne and heterodyne tomography*, Optics Express **30**, 15184–15200 (2022). https://doi.org/10.1364/OE.456597
- **CO:** I. Strandberg, *Simple, Reliable, and Noise-Resilient Continuous-Variable Quantum State Tomography with Convex Optimization*, Physical Review Applied **18**, 044041 (2022). https://doi.org/10.1103/PhysRevApplied.18.044041
- **CGAN:** S. Ahmed, C. Sánchez Muñoz, F. Nori, and A. F. Kockum, *Quantum State Tomography with Conditional Generative Adversarial Networks*, Physical Review Letters **127**, 140502 (2021). https://doi.org/10.1103/PhysRevLett.127.140502
- **Related nonparametric baseline source:** L. Markovich, X. Liu, and J. Tura, *Nonparametric Learning Non-Gaussian Quantum States of Continuous Variable Systems*, Physical Review Letters **137**, 020201 (2026). https://doi.org/10.1103/xdcg-6df5

The `.npy`/`.pkl` comparison results were received as separate numerical artifacts from prior computations. References to the publications do **not** by themselves establish permission to distribute the saved matrices or other people's implementation code. Those artifacts are omitted. The MIT license of this repository applies only to material that the authors can license.

The existing comparison notebooks perform numerical **postprocessing** of externally reconstructed states (including conversion between position and Fock representations); this is not equivalent to reproducing each original optimizer or machine-learning training procedure.
