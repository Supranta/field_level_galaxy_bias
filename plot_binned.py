"""plot_binned.py — Plot statistics vs density from per-bin fits.

Reads the per-bin summary pickles produced by fit_binned.py and generates
two figures:
    mean_variance_delta.png  Mean, variance/mean, and sigma vs log(1+delta).
    rho_c_delta.png          Pairwise cross-correlations vs log(1+delta).

Usage
-----
    python scripts/plot_binned.py <config.yaml>

See configs/binned_example.yaml for the expected config format.
"""
import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import trange

from lnp.data import load_data, compute_delta_bins, compute_delta_mean
from lnp.plotting import plot_mean_variance_sigma, plot_crosscorr_vs_density


def load_summaries(savedir, n_delta_bins):
    summaries = []
    for i in trange(n_delta_bins, desc='Loading summaries'):
        with open(savedir + '/%d.pkl' % i, 'rb') as f:
            summaries.append(pickle.load(f))
    return summaries


def stack(summaries, key):
    return np.array([s[key] for s in summaries])


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_delta_bins = cfg['n_delta_bins']
    datafile     = cfg['datafile']
    savedir      = cfg['savedir']
    catalog      = cfg['catalog']

    delta_slab, Ng = load_data(datafile, catalog)
    N_types    = Ng.shape[0]
    delta_flat = delta_slab.flatten()
    delta_bins = compute_delta_bins(delta_flat, n_delta_bins)
    delta_mean = compute_delta_mean(delta_flat, delta_bins)
    r_axis     = np.log(1.0 + delta_mean)

    summaries = load_summaries(savedir, n_delta_bins)

    data_mean   = stack(summaries, 'data_mean')    # (n_bins, N_types)
    data_var    = stack(summaries, 'data_var')
    model_mean  = stack(summaries, 'model_mean')
    model_var   = stack(summaries, 'model_var')
    data_rho_c  = stack(summaries, 'data_rho_c')  # (n_bins, N_types, N_types)
    model_rho_c = stack(summaries, 'model_rho_c')
    sigma_raw   = stack(summaries, 'sigma')        # (n_bins, n_samples, N_types)
    sigma_mean  = sigma_raw.mean(1)                # (n_bins, N_types)

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
        print("Usage: python scripts/plot_binned.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
