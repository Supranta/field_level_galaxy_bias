# Project: Lognormal-Poisson Galaxy Bias Model

## Overview
Models galaxy counts as a **lognormal-Poisson process** across multiple galaxy types in a matter density field. Goal: understand how galaxy **bias** (mean count vs density), **stochasticity** (variance/mean = Fano factor), and **cross-correlations** between types depend on local matter density.

## Repository Structure

```
lnp/                    Core library package
  data.py               load_data(), compute_delta_bins(), compute_delta_mean(), bin_mask()
  models.py             joint_lognormal_model (binned) + build_model() factory (density)
  density_functions.py  neyrinck_model_jax, sigma_model_jax
  inference.py          run_nuts(), posterior_predictive_binned(), compute_summaries()
  plotting.py           Shared plot helpers used by both pipelines

fit_binned.py           Non-parametric per-bin MCMC fit
fit_density.py          Parametric full-field MCMC fit
plot_binned.py          Figures for binned pipeline
plot_density_new.py     Figures for density pipeline (new version using lnp.*)

configs/
  binned_example.yaml   Example config for the binned pipeline
  density_example.yaml  Example config for the density pipeline
```

Old scripts (superseded but kept for reference): `fit_lognormal_poisson.py`, `fit_lognormal_poisson_density.py`, `plot.py`, `plot_density.py`, `models.py`, `density_functions.py`

Install the package before running scripts: `pip install -e .`

## Two Analysis Pipelines

### 1. Non-parametric binned (`fit_binned` + `plot_binned`)
Splits the field into `n_delta_bins` equal-count percentile bins of delta. Fits `joint_lognormal_model` independently per bin — rate and sigma are free constants per bin. Output: per-bin `.pkl` summary files.

### 2. Parametric density (`fit_density` + `plot_density`)
Fits a parametric model over all pixels simultaneously, with rate and sigma as explicit functions of r = 1 + delta. Output: single `samples.pkl` + `log_likelihood.npy`.

## Model Description

### joint_lognormal_model (binned pipeline)
- `sigma_t` ~ HalfNormal(1) — per-type lognormal scatter
- `rate_t` ~ Normal — per-type mean count
- `z_i` ~ Normal(0,1) — **shared** latent field per pixel (all types share same z)
- `lambda_i = exp(sigma_t * z_i - sigma_t^2 / 2)`
- `N_t_i ~ Poisson(rate_t * lambda_i)`

### build_model() factory (density pipeline)
Config-driven, three orthogonal choices:
- `mean_type`: `'neyrinck'` (`n_bar * r^beta * exp(-rho_g/r)`) or `'powerlaw'` (`n_bar * r^beta`)
- `z_type`: `'shared'`, `'independent'`, or `'zero'` (pure Poisson)
- `sigma_type`: `'density'` (`S * (r^gamma1 + A_sigma * r^gamma2)`) or `'constant'`

## Data Format
HDF5 file containing:
- `delta_2d`: 2D matter overdensity field
- `{catalog}/Ng`: galaxy counts, shape `(N_types, H, W)`

## Config YAML Keys

**Binned pipeline** (`fit_binned.py` / `plot_binned.py`):
```yaml
n_delta_bins: 10
datafile: /path/to/data.h5
savedir:  /path/to/output
catalog:  catalog_name
num_warmup:  500   # optional
num_samples: 500   # optional
```

**Density pipeline** (`fit_density.py` / `plot_density.py`):
```yaml
n_delta_bins: 10   # used only for plotting
datafile: /path/to/data.h5
savedir:  /path/to/output
catalog:  catalog_name
mean_type:  neyrinck
z_type:     shared
sigma_type: density
num_warmup:  500   # optional
num_samples: 500   # optional
```

## Outputs
- `<savedir>/<n>.pkl` — per-bin summaries: mean, var, rho_c, rate, sigma (binned)
- `<savedir>/samples.pkl` — MCMC posterior samples (density)
- `<savedir>/log_likelihood.npy` — per-sample log-likelihood (density)
- `<savedir>/figs/mean_variance_delta.png` — mean, variance/mean, sigma vs log(1+delta)
- `<savedir>/figs/rho_c_delta.png` — pairwise cross-correlations vs log(1+delta)
- `<savedir>/figs/pdf/<n>.png` — per-bin count PDFs (binned)
- `<savedir>/figs/corrcoef/<n>.png` — per-bin correlation matrices (binned)
