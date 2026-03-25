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
from lnp.plotting import (plot_mean_variance_sigma, plot_crosscorr_vs_density,
                          plot_getdist_contours, plot_latent_vs_density,
                          plot_latent_power_spectra, plot_combined_crosscorr_spectra,
                          plot_latent_maps)
from lnp.power_spectrum import (make_k_bins, compute_latent_power_spectra,
                                 compute_residual_crosscorr_spectra)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_delta_bins  = cfg['n_delta_bins']
    datafile      = cfg['datafile']
    savedir       = cfg['savedir']
    catalog       = cfg['catalog']
    fit_cfg       = cfg['fit_model']
    mean_type     = fit_cfg['mean_type']
    z_type        = fit_cfg['z_type']
    sigma_type    = fit_cfg.get('sigma_type')
    box_size      = cfg.get('box_size', 1000.)
    n_latent_bins = cfg.get('n_latent_bins', n_delta_bins)
    n_k_bins      = cfg.get('n_k_bins', 20)

    HAS_Z             = z_type != 'zero'
    HAS_DENSITY_SIGMA = HAS_Z and sigma_type == 'density'
    HAS_NEYRINCK_MEAN = mean_type == 'neyrinck'

    delta_slab, Ng = load_data(datafile, catalog)
    N_slabs, H, W  = delta_slab.shape
    assert H == W, "Field2D requires a square grid; got H=%d, W=%d" % (H, W)
    N_grid     = H
    N_types    = Ng.shape[0]
    delta_flat = delta_slab.flatten()   # unsmoothed — always used for bin x-axis
    Ng_flat    = Ng.reshape(N_types, -1)

    with open(savedir + '/samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    samples = {k: np.array(v) for k, v in samples.items()}
    n_mcmc  = samples['n_bar'].shape[0]
    is_map  = n_mcmc == 1

    print("Model:   mean=%s  z=%s  sigma=%s" % (mean_type, z_type, sigma_type))
    if is_map:
        print("Mode:    MAP (single point estimate — posterior uncertainty unavailable)")

    r_flat = 1.0 + delta_flat
    print("N_types: %d,  n_mcmc: %d" % (N_types, n_mcmc))

    delta_bins = compute_delta_bins(delta_flat, n_delta_bins)
    delta_mean = compute_delta_mean(delta_flat, delta_bins)
    r_axis     = np.log(1.0 + delta_mean)

    # ---- Model-aware posterior predictive helpers ----

    def _mu_det(r_3d):
        """Deterministic mean rate. Shape: (n_mcmc, N_types, N_pix_bin)."""
        if HAS_NEYRINCK_MEAN:
            dg = samples['delta_g']
            # neyrinck_shared: dg is (n_mcmc,); neyrinck: dg is (n_mcmc, N_types)
            if dg.ndim == 1:
                dg = dg[:, None, None]   # broadcast over types and pixels
            else:
                dg = dg[:, :, None]
            return np.array(neyrinck_model_jax(
                r_3d,
                samples['n_bar'][:, :, None],
                samples['beta'][:, :, None],
                dg,
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
        Ng_bin = Ng_flat[:, select_mask]
        r_bin  = r_flat[select_mask].astype(np.float32)
        r_mean = float(r_flat[select_mask].mean())
        r_3d   = r_bin[None, None, :]                  # (1, 1, N_pix_bin)

        mu = _mu_det(r_3d)                             # (n_mcmc, N_types, N_pix_bin)

        if HAS_Z:
            z_bin = samples['z'][:, select_mask]
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

    # ---- Plot 1: GetDist parameter contours ----
    # Skipped for MAP runs: GetDist requires multiple samples to estimate a
    # posterior distribution; a point estimate carries no uncertainty to plot.
    if is_map:
        print("Skipping parameter contours (MAP mode — no posterior distribution).")
    else:
        print("Plotting parameter contours...")
        names, labels, columns = [], [], []

        per_type_params = ['n_bar', 'beta']
        per_type_latex  = [r'\bar{n}', r'\beta']
        if mean_type == 'neyrinck':
            per_type_params += ['delta_g']
            per_type_latex  += [r'\delta_g']
        if HAS_Z:
            if HAS_DENSITY_SIGMA:
                per_type_params += ['S']
                per_type_latex  += ['S']
            else:
                per_type_params += ['sigma']
                per_type_latex  += [r'\sigma']

        for key, latex in zip(per_type_params, per_type_latex):
            for t in range(N_types):
                names.append('%s^%d' % (key, t))
                labels.append('%s^{%d}' % (latex, t))
                columns.append(samples[key][:, t])

        if mean_type == 'neyrinck_shared':
            names.append('delta_g')
            labels.append(r'\delta_g')
            columns.append(samples['delta_g'])

        if HAS_Z and HAS_DENSITY_SIGMA:
            for key, latex in [('gamma1', r'\gamma_1'), ('gamma2', r'\gamma_2'),
                               ('A_sigma', 'A_\\sigma')]:
                names.append(key)
                labels.append(latex)
                columns.append(samples[key])

        data_matrix = np.column_stack(columns)
        savepath = savedir + '/figs/param_contours.png'
        plot_getdist_contours(data_matrix, names, labels, savepath)
        print("Saved:", savepath)

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

    # ---- Plot 2: mean / variance / sigma ----
    print("Plotting mean / variance / sigma vs density...")
    fig, ax = plt.subplots(N_types, 3, figsize=(13., 3 * N_types))
    plot_mean_variance_sigma(ax, r_axis, data_mean, data_vom,
                             model_mean, model_vom, sigma_mean)
    plt.tight_layout()
    savepath = savedir + '/figs/mean_variance_delta.png'
    plt.savefig(savepath, dpi=150.)
    plt.close()
    print("Saved:", savepath)

    # ---- Plot 3: pairwise cross-correlations ----
    print("Plotting cross-correlations vs density...")
    fig, ax = plt.subplots(N_types - 1, N_types - 1,
                           figsize=((N_types - 1) * 4., (N_types - 1) * 3.))
    plot_crosscorr_vs_density(ax, r_axis, data_rho_c, model_rho_c)
    plt.tight_layout()
    savepath = savedir + '/figs/rho_c_delta.png'
    plt.savefig(savepath, dpi=150.)
    plt.close()
    print("Saved:", savepath)

    # ---- Plots 4–7: latent field diagnostics (only when latent z exists) ----
    def _plot_latent_diagnostics():
        """Compute and save four diagnostic figures for the posterior latent z field."""
        z_samples = samples['z']   # (n_mcmc, N_pix)

        # -- Plot 4: z mean and std vs delta --
        latent_bins       = compute_delta_bins(delta_flat, n_latent_bins)
        latent_delta_mean = compute_delta_mean(delta_flat, latent_bins)
        r_latent_axis     = np.log(1.0 + latent_delta_mean)

        z_mean_arr = np.empty(n_latent_bins)
        z_std_arr  = np.empty(n_latent_bins)
        n_pix_arr  = np.empty(n_latent_bins, dtype=int)
        for b in range(n_latent_bins):
            mask          = (delta_flat > latent_bins[b]) & (delta_flat <= latent_bins[b + 1])
            z_in_bin      = z_samples[:, mask]        # (n_mcmc, n_pix_bin)
            z_mean_arr[b] = z_in_bin.mean(0).mean()   # pixel-avg then sample-avg
            z_std_arr[b]  = z_in_bin.std(0).mean() if not is_map else np.nan
            n_pix_arr[b]  = mask.sum()

        fig, ax = plt.subplots(1, 2, figsize=(10., 3.5))
        plt.suptitle("Latent field z vs density")
        plot_latent_vs_density(ax, r_latent_axis, z_mean_arr, z_std_arr, n_pix_arr)
        plt.tight_layout()
        savepath = savedir + '/figs/latent_vs_density.png'
        plt.savefig(savepath, dpi=150.)
        plt.close()
        print("Saved:", savepath)

        # -- Spectral quantities (expensive per-sample loop) --
        print("Computing latent field power spectra...")
        (k_centres,
         log_pk_mean, log_pk_std,
         log_pk_prior_mean, log_pk_prior_std,
         rho_c_mean, rho_c_std) = compute_latent_power_spectra(
            z_samples, delta_slab, N_grid,
            box_size=box_size, n_k_bins=n_k_bins,
        )

        # -- Residual cross-correlation with delta --
        # For each MCMC sample: compute rate_i, form residual_i = rate_i - Ng,
        # compute rho_c(k) for that sample, accumulate. This mirrors how the
        # latent spectra are computed and yields mean + uncertainty.
        print("Computing per-sample residual cross-correlation spectra...")
        N_pix              = delta_flat.shape[0]
        r_3d               = r_flat[None, None, :]   # (1, 1, N_pix)
        field, k_bins      = make_k_bins(N_grid, box_size, n_k_bins)
        rho_c_resid_samples = np.empty((n_mcmc, N_types, N_slabs, n_k_bins))

        for i in trange(n_mcmc, desc="Residual spectra"):
            si = {k: v[i:i+1] for k, v in samples.items()}

            if HAS_NEYRINCK_MEAN:
                dg = si['delta_g']
                dg = dg[:, None, None] if dg.ndim == 1 else dg[:, :, None]
                mu = np.array(neyrinck_model_jax(
                    r_3d, si['n_bar'][:, :, None], si['beta'][:, :, None], dg
                ))
            else:
                mu = si['n_bar'][:, :, None] * r_3d ** si['beta'][:, :, None]

            if HAS_DENSITY_SIGMA:
                sigma = np.array(sigma_model_jax(
                    r_3d,
                    si['S'][:, :, None],
                    si['gamma1'][:, None, None],
                    si['gamma2'][:, None, None],
                    si['A_sigma'][:, None, None],
                ))
            else:
                sigma = si['sigma'][:, :, None] * np.ones((1, 1, N_pix))

            # mu, sigma both (1, N_types, N_pix); z_samples[i] is (N_pix,)
            lam        = np.exp(sigma[0] * z_samples[i] - 0.5 * sigma[0] ** 2)
            rate_i     = mu[0] * lam                              # (N_types, N_pix)
            residual_i = Ng_flat - rate_i                        # (N_types, N_pix)
            resid_map_i = residual_i.reshape(N_types, N_slabs, N_grid, N_grid)

            _, rho_c_i = compute_residual_crosscorr_spectra(
                resid_map_i, delta_slab, field, k_bins
            )
            rho_c_resid_samples[i] = rho_c_i

        rho_c_resid_mean = rho_c_resid_samples.mean(0)   # (N_types, N_slabs, n_k_bins)
        rho_c_resid_std  = rho_c_resid_samples.std(0)

        n_show = min(8, N_slabs)
        n_rows = (n_show + 3) // 4   # 1 or 2 rows of 4 panels

        # -- Plot 5: per-slab latent power spectra --
        fig, ax = plt.subplots(n_rows, 4, figsize=(10., 5. * n_rows))
        plot_latent_power_spectra(ax, k_centres,
                                  log_pk_mean[:n_show], log_pk_std[:n_show],
                                  log_pk_prior_mean, log_pk_prior_std)
        plt.tight_layout()
        savepath = savedir + '/figs/latent_power_spectra.png'
        plt.savefig(savepath, dpi=150.)
        plt.close()
        print("Saved:", savepath)

        # -- Plot 6: per-slab cross-correlation with delta (latent + residual) --
        fig, ax = plt.subplots(n_rows, 4, figsize=(14., 5. * n_rows))
        plot_combined_crosscorr_spectra(
            ax, k_centres,
            rho_c_mean[:n_show], rho_c_std[:n_show],
            rho_c_resid_mean=rho_c_resid_mean[:, :n_show, :],
            rho_c_resid_std=rho_c_resid_std[:, :n_show, :],
        )
        plt.tight_layout()
        savepath = savedir + '/figs/latent_crosscorr_spectra.png'
        plt.savefig(savepath, dpi=150.)
        plt.close()
        print("Saved:", savepath)

        # -- Plot 7: 2D map images of delta, mean z, std z --
        z_mean_map  = z_samples.mean(0).reshape(N_slabs, H, W)
        z_std_map   = z_samples.std(0).reshape(N_slabs, H, W)
        n_show_maps = min(4, N_slabs)
        fig, ax = plt.subplots(3, n_show_maps, figsize=(10., 8.5))
        plot_latent_maps(ax, delta_slab, z_mean_map, z_std_map, n_show=n_show_maps)
        plt.tight_layout()
        savepath = savedir + '/figs/latent_maps.png'
        plt.savefig(savepath, dpi=150.)
        plt.close()
        print("Saved:", savepath)

    if HAS_Z:
        print("Plotting latent field diagnostics...")
        _plot_latent_diagnostics()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_density.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
