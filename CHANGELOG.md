# Changelog

## v1.1 — 2026-09-18

First public VIML release of the Fokker–Planck surrogate workflow.

### Included
- Conservative 1D Chang–Cooper Fokker–Planck solver with zero-flux boundaries.
- Sobol-sampled parameterized trajectory generation (5,000 trajectories by default).
- PCA compression and a 16-component latent representation.
- Direct MLP baseline.
- Initial-condition-aware residual MLP:
  \[
  z(t)=z_0(\sigma_0)+\frac{t}{t_{\max}}g(D_0,\nu_{\rm coll},\sigma_0,t).
  \]
- Held-out error analysis and representative solver-vs-surrogate figure.
- Lightweight pretrained residual checkpoint and inference CLI.

### Repository cleanup
- Portable module imports and shell wrappers.
- Generated datasets and large test-prediction arrays excluded from version control.
- Added README, requirements, citation metadata, version marker, and VIML license.
