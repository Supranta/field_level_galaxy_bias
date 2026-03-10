# Lognormal-Poisson Galaxy Bias Model

Models galaxy counts as a **lognormal-Poisson process** across multiple galaxy
types in a matter density field. The goal is to characterise how galaxy
**bias** (mean count vs density), **stochasticity** (variance/mean), and
**cross-correlations** between types depend on local matter density.

## Model overview

Each pixel $i$ has a local matter density $\delta_i$, and each galaxy type $t$
has a count $N_t^i$ drawn from:

$$
N_t^i \sim \mathrm{Poisson}\!\left(\mu_t(\delta_i) \cdot \lambda_t^i\right)
$$

where:
- $\mu_t(\delta)$ is the **mean bias function** (Neyrinck or power-law),
- $\lambda_t^i = \exp(\sigma_t \cdot z_i - \tfrac{1}{2}\sigma_t^2)$ is a
  **lognormal scatter** factor with mean 1, and
- $z_i \sim \mathcal{N}(0, 1)$ is a latent Gaussian field.

When all types share the same $z_i$ (the default), their counts are positively
correlated. The scatter $\sigma_t$ can be a constant or a function of density.

## Two analysis pipelines

### 1. Non-parametric binned fit (`fit_binned` + `plot_binned`)

The density field is split into equal-count bins of $\delta$. A simple
shared-$z$ lognormal-Poisson model is fit **independently** within each bin,
treating rate and $\sigma$ as free constants. This gives a non-parametric
picture of how bias and stochasticity vary with density.

### 2. Parametric density fit (`fit_density` + `plot_density`)

A single MCMC run fits a parametric model over **all pixels** simultaneously,
with rate and $\sigma$ as explicit functions of $r = 1 + \delta$:

| `mean_type`  | Mean function $\mu(r)$                          |
|--------------|-------------------------------------------------|
| `neyrinck`   | $\bar{n}\, r^\beta \exp(-\rho_g / r)$           |
| `powerlaw`   | $\bar{n}\, r^\beta$                             |

| `sigma_type` | Scatter $\sigma(r)$                             |
|--------------|-------------------------------------------------|
| `density`    | $S\,(r^{\gamma_1} + A_\sigma\, r^{\gamma_2})$   |
| `constant`   | Per-type constant                               |

Set `z_type: zero` for a pure Poisson model (no lognormal scatter).

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # install the lnp package in editable mode
```

## Data format

An HDF5 file with:
- `delta_2d` : `(H, W)` 2D matter overdensity field.
- `<catalog>/Ng` : `(N_types, H, W)` galaxy counts per type.

## Usage

**Non-parametric binned pipeline:**

```bash
python fit_binned.py  configs/my_binned.yaml   # run MCMC per bin
python plot_binned.py configs/my_binned.yaml   # generate figures
```

**Parametric density pipeline:**

```bash
python fit_density.py      configs/my_density.yaml  # run single MCMC
python plot_density.py configs/my_density.yaml  # generate figures
```

## Output files

| Path | Description |
|------|-------------|
| `<savedir>/<n>.pkl` | Per-bin summary statistics (binned pipeline) |
| `<savedir>/samples.pkl` | MCMC posterior samples (density pipeline) |
| `<savedir>/log_likelihood.npy` | Per-sample log-likelihood (density pipeline) |
| `<savedir>/figs/mean_variance_delta.png` | Mean, variance/mean, $\sigma$ vs $\log(1+\delta)$ |
| `<savedir>/figs/rho_c_delta.png` | Cross-correlations vs $\log(1+\delta)$ |
| `<savedir>/figs/param_contours.png` | GetDist triangle plot of global parameters (density pipeline) |
| `<savedir>/figs/pdf/<n>.png` | Per-bin count PDFs (binned pipeline) |
| `<savedir>/figs/corrcoef/<n>.png` | Per-bin correlation matrices (binned pipeline) |

## Repository structure

```
lnp/                    Core library
  data.py               Data loading and density binning utilities
  models.py             NumPyro models (joint_lognormal_model, build_model)
  density_functions.py  Functional forms: Neyrinck mean, two-power-law sigma
  inference.py          NUTS runner, posterior predictive, summary statistics
  plotting.py           Shared plotting functions

fit_binned.py           Non-parametric per-bin MCMC fit
fit_density.py          Parametric full-field MCMC fit
plot_binned.py          Figures for binned pipeline
plot_density.py     Figures for density pipeline

configs/
  binned_example.yaml   Example config for the binned pipeline
  density_example.yaml  Example config for the density pipeline
```
