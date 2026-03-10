"""plot_density.py — Plot statistics vs density from a parametric density fit.

Reads samples.pkl produced by fit_density.py and generates two figures:
    mean_variance_delta.png  Mean, variance/mean, and sigma vs log(1+delta).
    rho_c_delta.png          Pairwise cross-correlations vs log(1+delta).

Usage
-----
    python scripts/plot_density.py <config.yaml>

See configs/density_example.yaml for the expected config format.
"""
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import trange

from lnp.data import load_data, compute_delta_bins, compute_delta_mean
from lnp.density_functions import neyrinck_model_jax, sigma_model_jax
from lnp.plotting import plot_mean_variance_sigma, plot_crosscorr_vs_density


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_delta_bins = cfg['n_delta_bins']
    datafile     = cfg['datafile']
    savedir      = cfg['savedir']
    catalog      = cfg['catalog']
    mean_type    = cfg['mean_type']
    z_type       = cfg['z_type']
    sigma_type   = cfg.get('sigma_type')

    HAS_Z             = z_type != 'zero'
    IS_INDEP_Z        = z_type == 'independent'
    HAS_DENSITY_SIGMA = HAS_Z and sigma_type == 'density'
    HAS_NEYRINCK_MEAN = mean_type == 'neyrinck'

    delta_slab, Ng = load_data(datafile, catalog)
    N_types    = Ng.shape[0]
    delta_flat = delta_slab.flatten()
    Ng_flat    = Ng.reshape(N_types, -1)

    with open(savedir + '/samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    samples = {k: np.array(v) for k, v in samples.items()}
    n_mcmc  = samples['n_bar'].shape[0]

    print("Model:   mean=%s  z=%s  sigma=%s" % (mean_type, z_type, sigma_type))
    print("N_types: %d,  n_mcmc: %d" % (N_types, n_mcmc))

    delta_bins = compute_delta_bins(delta_flat, n_delta_bins)
    delta_mean = compute_delta_mean(delta_flat, delta_bins)
    r_axis     = np.log(1.0 + delta_mean)

    # ---- Model-aware posterior predictive helpers ----

    def _mu_det(r_3d):
        """Deterministic mean rate. Shape: (n_mcmc, N_types, N_pix_bin)."""
        if HAS_NEYRINCK_MEAN:
            return np.array(neyrinck_model_jax(
                r_3d,
                samples['n_bar'][:, :, None],
                samples['beta'][:, :, None],
                samples['delta_g'][:, :, None],
            ))
        return samples['n_bar'][:, :, None] * r_3d ** samples['beta'][:, :, None]

    def _sigma_arr(r_3d):
        """Sigma array. Shape: (n_mcmc, N_types, N_pix_bin). None for pure Poisson."""
        if not HAS_Z:
            return None
        if HAS_DENSITY_SIGMA:
            return np.array(sigma_model_jax(
                r_3d,
                samples['S'][:, :, None],
                samples['gamma1'][:, None, None],
                samples['gamma2'][:, None, None],
                samples['A_sigma'][:, None, None],
            ))
        N_pix_bin = r_3d.shape[2]
        return samples['sigma'][:, :, None] * np.ones((1, 1, N_pix_bin))

    def _lam(sigma, z_bin):
        if IS_INDEP_Z:
            return np.exp(sigma * z_bin - 0.5 * sigma ** 2)
        return np.exp(sigma * z_bin[:, None, :] - 0.5 * sigma ** 2)

    def _sigma_at_mean(r_mean):
        """Posterior-mean sigma evaluated at r_mean. Shape: (N_types,). None for pure Poisson."""
        if not HAS_Z:
            return None
        if HAS_DENSITY_SIGMA:
            return np.array(sigma_model_jax(
                r_mean,
                samples['S'],
                samples['gamma1'][:, None],
                samples['gamma2'][:, None],
                samples['A_sigma'][:, None],
            )).mean(0)
        return samples['sigma'].mean(0)

    def get_bin_summaries(select_mask):
        Ng_bin    = Ng_flat[:, select_mask]
        delta_bin = delta_flat[select_mask]
        r_bin     = (1.0 + delta_bin).astype(np.float32)
        r_mean    = float(1.0 + delta_bin.mean())
        r_3d      = r_bin[None, None, :]               # (1, 1, N_pix_bin)

        mu = _mu_det(r_3d)                             # (n_mcmc, N_types, N_pix_bin)

        if HAS_Z:
            z_bin = (samples['z'][:, :, select_mask]
                     if IS_INDEP_Z
                     else samples['z'][:, select_mask])
            sigma = _sigma_arr(r_3d)
            rate  = mu * _lam(sigma, z_bin)
        else:
            rate = mu

        Ng_pp      = np.random.poisson(rate)           # (n_mcmc, N_types, N_pix_bin)
        Ng_pp_flat = Ng_pp.transpose(1, 0, 2).reshape(N_types, -1)

        return {
            'data_mean':   Ng_bin.mean(1),
            'data_var':    Ng_bin.var(1),
            'data_rho_c':  np.corrcoef(Ng_bin),
            'model_mean':  Ng_pp_flat.mean(1),
            'model_var':   Ng_pp_flat.var(1),
            'model_rho_c': np.corrcoef(Ng_pp_flat),
            'sigma_mean':  _sigma_at_mean(r_mean),
        }

    # ---- Per-bin summaries ----
    print("Computing per-bin summaries...")
    bin_summaries = []
    for n in trange(n_delta_bins):
        select = (delta_flat > delta_bins[n]) & (delta_flat <= delta_bins[n + 1])
        bin_summaries.append(get_bin_summaries(select))

    def stack(key):
        return np.array([s[key] for s in bin_summaries])

    data_mean   = stack('data_mean')
    data_var    = stack('data_var')
    model_mean  = stack('model_mean')
    model_var   = stack('model_var')
    data_rho_c  = stack('data_rho_c')
    model_rho_c = stack('model_rho_c')
    sigma_mean  = stack('sigma_mean') if HAS_Z else None

    data_vom  = data_var  / data_mean
    model_vom = model_var / model_mean

    os.makedirs(savedir + '/figs', exist_ok=True)

    # ---- Plot 1: mean / variance / sigma ----
    print("Plotting mean / variance / sigma vs density...")
    fig, ax = plt.subplots(N_types, 3, figsize=(13., 3 * N_types))
    plot_mean_variance_sigma(ax, r_axis, data_mean, data_vom,
                             model_mean, model_vom, sigma_mean)
    plt.tight_layout()
    savepath = savedir + '/figs/mean_variance_delta.png'
    plt.savefig(savepath, dpi=150.)
    plt.close()
    print("Saved:", savepath)

    # ---- Plot 2: pairwise cross-correlations ----
    print("Plotting cross-correlations vs density...")
    fig, ax = plt.subplots(N_types - 1, N_types - 1,
                           figsize=((N_types - 1) * 4., (N_types - 1) * 3.))
    plot_crosscorr_vs_density(ax, r_axis, data_rho_c, model_rho_c)
    plt.tight_layout()
    savepath = savedir + '/figs/rho_c_delta.png'
    plt.savefig(savepath, dpi=150.)
    plt.close()
    print("Saved:", savepath)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_density.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
