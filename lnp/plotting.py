import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib import cm
from getdist import MCSamples
import getdist.plots as gdplots


def plot_hist(ax, Ng_data, Ng_model):
    """Overlay data and model count histograms on a single Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    Ng_data  : (N_pix,) array
    Ng_model : (N_pp,) array
    """
    bins = np.arange(Ng_data.min(), Ng_data.max() + 1)
    ax.hist(Ng_data,  bins=bins, histtype='bar',  color='gray', density=True, label='Data')
    ax.hist(Ng_model, bins=bins, histtype='step', density=True, label='Model')


def plot_mean_variance_sigma(ax, r_axis, data_mean, data_vom,
                             model_mean, model_vom, sigma_mean=None):
    """Fill a (N_types, 3) axes grid with mean, variance/mean, and sigma vs log(1+delta).

    Parameters
    ----------
    ax : (N_types, 3) array of Axes
    r_axis : (n_bins,) ndarray
        log(1 + delta_mean) per bin.
    data_mean, model_mean : (n_bins, N_types) ndarray
    data_vom, model_vom : (n_bins, N_types) ndarray
        Variance / mean per bin.
    sigma_mean : (n_bins, N_types) ndarray or None
        Posterior-mean sigma per bin. If None, the sigma panel shows
        'N/A (pure Poisson)'.
    """
    N_types = data_mean.shape[1]
    for i in range(N_types):
        for j in range(3):
            ax[i, j].set_xlabel(r'$\log[1 + \delta]$')

        ax[i, 0].set_ylabel(r'$N_g$')
        ax[i, 0].set_title('Type %d' % (i + 1))
        ax[i, 0].plot(r_axis, data_mean[:, i],  'ks-', markerfacecolor='white', label='Data')
        ax[i, 0].plot(r_axis, model_mean[:, i], 'r--', label='Model')

        ax[i, 1].set_ylabel('Variance / Mean')
        ax[i, 1].set_title('Type %d' % (i + 1))
        ax[i, 1].plot(r_axis, data_vom[:, i],  'ks-', markerfacecolor='white', label='Data')
        ax[i, 1].plot(r_axis, model_vom[:, i], 'r--', label='Model')

        ax[i, 2].set_ylabel(r'$\sigma$')
        ax[i, 2].set_title('Type %d' % (i + 1))
        if sigma_mean is not None:
            ax[i, 2].plot(r_axis, sigma_mean[:, i], 'ro-', markerfacecolor='white', label='Model')
        else:
            ax[i, 2].text(0.5, 0.5, 'N/A\n(pure Poisson)',
                          ha='center', va='center',
                          transform=ax[i, 2].transAxes, fontsize=12, color='gray')

    ax[0, 0].legend()


def plot_crosscorr_vs_density(ax, r_axis, data_rho_c, model_rho_c):
    """Fill a lower-triangular (N_types-1, N_types-1) axes grid with
    pairwise cross-correlations vs log(1+delta).

    Parameters
    ----------
    ax : (N_types-1, N_types-1) array of Axes, or a single Axes for N_types=2
    r_axis : (n_bins,) ndarray
    data_rho_c, model_rho_c : (n_bins, N_types, N_types) ndarray
    """
    N_types = data_rho_c.shape[1]
    for i in range(N_types - 1):
        for j in range(N_types - 1):
            ax_ij = ax[i, j] if N_types > 2 else ax
            if j > i:
                ax_ij.axis('off')
            else:
                ax_ij.set_ylim(0., 1.)
                ax_ij.set_title('Type %d\u2013Type %d' % (j + 1, i + 2))
                ax_ij.set_xlabel(r'$\log[1 + \delta]$')
                ax_ij.set_ylabel(r'$\rho_c$')
                ax_ij.plot(r_axis, data_rho_c[:, i + 1, j],
                           'ks-', markerfacecolor='white', label='Data')
                ax_ij.plot(r_axis, model_rho_c[:, i + 1, j], 'r--', label='Model')

    if N_types > 2:
        ax[0, 0].legend()
    else:
        ax.legend()


def plot_count_pdfs(Ng_data, Ng_model, N_types, savepath):
    """Plot per-type count histograms (data vs model) and save to file.

    Parameters
    ----------
    Ng_data  : (N_types, N_pix) ndarray
    Ng_model : (N_types, N_pp)  ndarray
    N_types  : int
    savepath : str
    """
    fig, ax = plt.subplots(N_types, 2, figsize=(8., 2.5 * N_types))
    for j in range(N_types):
        for i in range(2):
            ax[j, i].set_title('Type %d' % (j + 1))
            ax[j, i].set_xlabel(r'$N_g$')
            plot_hist(ax[j, i], Ng_data[j], Ng_model[j])
        ax[j, 1].set_yscale('log')
    ax[0, 0].legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=150.)
    plt.close()


def plot_getdist_contours(data_matrix, names, labels, savepath):
    """Triangle (corner) plot of global model parameters using GetDist.

    Parameters
    ----------
    data_matrix : (n_mcmc, n_params) ndarray
    names       : list of str  — plain-string parameter names (no spaces)
    labels      : list of str  — LaTeX labels (without $ delimiters)
    savepath    : str
    """
    mc_samples = MCSamples(samples=data_matrix, names=names, labels=labels)
    g = gdplots.get_subplot_plotter()
    g.triangle_plot([mc_samples], filled=True)
    g.export(savepath)
    plt.close('all')


def _normalize(x):
    return (x - x.mean()) / x.std()


def plot_latent_vs_density(ax, r_axis, z_mean_arr, z_std_arr, n_pix_arr):
    """Two-panel plot: posterior mean z and sigma_z vs log(1+delta).

    Parameters
    ----------
    ax          : (2,) array of Axes
    r_axis      : (n_bins,) ndarray — log(1 + delta_mean) per bin
    z_mean_arr  : (n_bins,) ndarray — pixel-avg posterior mean of z per bin
    z_std_arr   : (n_bins,) ndarray — pixel-avg posterior std dev of z per bin
    n_pix_arr   : (n_bins,) int array — pixel count per bin (for std-of-std error bars)
    """
    ax[0].set_ylabel(r'$\bar{z}$')
    ax[0].set_xlabel(r'$\log(1 + \delta)$')
    ax[0].errorbar(r_axis, z_mean_arr, z_std_arr,
                   color='k', fmt='o', markerfacecolor='white', capsize=5.)
    ax[0].axhline( 1., color='k', ls=':')
    ax[0].axhline(-1., color='k', ls=':')

    ax[1].set_ylabel(r'$\sigma_z$')
    ax[1].set_xlabel(r'$\log(1 + \delta)$')
    ax[1].axhline(1., color='r', ls='--', label='Prior std')
    ax[1].errorbar(r_axis, z_std_arr, z_std_arr / np.sqrt(n_pix_arr),
                   color='k', fmt='o', markerfacecolor='white', capsize=5.)
    ax[1].legend()


def plot_latent_power_spectra(ax, k_centres,
                              log_pk_mean, log_pk_std,
                              log_pk_prior_mean, log_pk_prior_std):
    """Fill a (n_rows, 4) axes grid with per-slab log P(k) vs k.

    Posterior is shown in red with shaded 1-sigma band; white-noise prior in blue.

    Parameters
    ----------
    ax                : (n_rows, 4) array of Axes (or (4,) for a single row)
    k_centres         : (n_k_bins,) ndarray
    log_pk_mean       : (N_slabs, n_k_bins) ndarray — posterior mean
    log_pk_std        : (N_slabs, n_k_bins) ndarray
    log_pk_prior_mean : (n_k_bins,) ndarray — white-noise reference
    log_pk_prior_std  : (n_k_bins,) ndarray
    """
    ax_flat = np.array(ax).flatten()
    N_slabs = log_pk_mean.shape[0]
    for s in range(N_slabs):
        a = ax_flat[s]
        a.set_title('Slab %d' % (s + 1))
        a.set_ylabel(r'$\ln P(k)$')
        a.set_xlabel(r'$k$')
        a.semilogx(k_centres, log_pk_mean[s], color='r', label='Posterior')
        a.fill_between(k_centres,
                       log_pk_mean[s] - log_pk_std[s],
                       log_pk_mean[s] + log_pk_std[s],
                       color='r', alpha=0.3)
        a.semilogx(k_centres, log_pk_prior_mean, color='b', label='Prior')
        a.fill_between(k_centres,
                       log_pk_prior_mean - log_pk_prior_std,
                       log_pk_prior_mean + log_pk_prior_std,
                       color='b', alpha=0.3)
    ax_flat[0].legend()

    # Hide unused panels
    for s in range(N_slabs, len(ax_flat)):
        ax_flat[s].axis('off')


def plot_combined_crosscorr_spectra(ax, k_centres, rho_c_mean, rho_c_std,
                                    rho_c_resid_mean=None, rho_c_resid_std=None,
                                    cmap_name='cividis'):
    """Fill a (n_rows, 4) axes grid with per-slab rho_c(k) diagnostics.

    Each panel shows:
      - Latent-delta rho_c: posterior mean as a red line with shaded 1-sigma band.
      - Residual-delta rho_c: one line + shaded band per galaxy type, coloured by
        a continuous colormap (optional).

    Parameters
    ----------
    ax               : (n_rows, 4) array of Axes (or (4,) for a single row)
    k_centres        : (n_k_bins,) ndarray
    rho_c_mean       : (N_slabs, n_k_bins) ndarray  — latent posterior mean
    rho_c_std        : (N_slabs, n_k_bins) ndarray  — latent posterior std
    rho_c_resid_mean : (N_types, N_slabs, n_k_bins) ndarray or None
    rho_c_resid_std  : (N_types, N_slabs, n_k_bins) ndarray or None
    cmap_name        : str  — matplotlib colormap for galaxy types
    """
    ax_flat = np.array(ax).flatten()
    N_slabs = rho_c_mean.shape[0]
    HAS_RESID = rho_c_resid_mean is not None
    N_types   = rho_c_resid_mean.shape[0] if HAS_RESID else 0
    cmap      = mcm.get_cmap(cmap_name)
    colors    = [cmap(t / max(N_types + 1, 1)) for t in range(N_types)]

    for s in range(N_slabs):
        a = ax_flat[s]
        a.set_title('Slab %d' % (s + 1))
        a.set_ylabel(r'$\rho_c(k)$')
        a.set_xlabel(r'$k$')
        a.set_ylim(-1.,1.)

        a.semilogx(k_centres, rho_c_mean[s], color='r', label='Latent z')
        a.fill_between(k_centres,
                       rho_c_mean[s] - rho_c_std[s],
                       rho_c_mean[s] + rho_c_std[s],
                       color='r', alpha=0.25)

        if HAS_RESID:
            for t in range(N_types):
                a.semilogx(k_centres, rho_c_resid_mean[t, s],
                           color=colors[t], label='Type %d residual' % (t + 1))
                if rho_c_resid_std is not None:
                    a.fill_between(
                        k_centres,
                        rho_c_resid_mean[t, s] - rho_c_resid_std[t, s],
                        rho_c_resid_mean[t, s] + rho_c_resid_std[t, s],
                        color=colors[t], alpha=0.2,
                    )

    ax_flat[0].legend(fontsize=7)

    for s in range(N_slabs, len(ax_flat)):
        ax_flat[s].axis('off')


def plot_latent_crosscorr_spectra(ax, k_centres, rho_c_mean, rho_c_std):
    """Fill a (n_rows, 4) axes grid with per-slab rho_c(k) between z and delta.

    Parameters
    ----------
    ax         : (n_rows, 4) array of Axes (or (4,) for a single row)
    k_centres  : (n_k_bins,) ndarray
    rho_c_mean : (N_slabs, n_k_bins) ndarray
    rho_c_std  : (N_slabs, n_k_bins) ndarray
    """
    plot_combined_crosscorr_spectra(ax, k_centres, rho_c_mean, rho_c_std)


def plot_latent_maps(ax, delta_slab, z_mean_map, z_std_map, n_show=4):
    """Fill a (3, n_show) axes grid with delta, mean-z, and std-z map images.

    Row 0: matter overdensity delta.
    Row 1: posterior mean of the latent field z.
    Row 2: posterior std dev of the latent field z.

    Parameters
    ----------
    ax          : (3, n_show) array of Axes
    delta_slab  : (N_slabs, H, W) ndarray
    z_mean_map  : (N_slabs, H, W) ndarray — MCMC mean of z
    z_std_map   : (N_slabs, H, W) ndarray — MCMC std of z
    n_show      : int — number of slabs to display (default 4)
    """
    for i in range(n_show):
        ax[0, i].set_title(r'$\delta$ (Slab %d)' % (i + 1))
        ax[1, i].set_title(r'$\langle z \rangle$ (Slab %d)' % (i + 1))
        ax[2, i].set_title(r'$\sigma_z$ (Slab %d)' % (i + 1))

        for row in range(3):
            ax[row, i].set_xticks([])
            ax[row, i].set_yticks([])

        ax[0, i].imshow(_normalize(delta_slab[i]),  vmin=-1.5, vmax=2.5)
        ax[1, i].imshow(_normalize(z_mean_map[i]),  vmin=-1.5, vmax=1.5)
        ax[2, i].imshow(z_std_map[i], vmin=0.1, vmax=0.5)


def plot_corrcoef_matrix(rho_c_data, rho_c_model, N_types, savepath):
    """Plot side-by-side correlation matrices (data vs model) and save to file.

    Parameters
    ----------
    rho_c_data, rho_c_model : (N_types, N_types) ndarray
    N_types  : int
    savepath : str
    """
    ticks = ['Type %d' % (k + 1) for k in range(N_types)]
    fig, ax = plt.subplots(1, 2, figsize=(10., 10.))
    for k in range(2):
        ax[k].set_xticks(np.arange(N_types), ticks)
        ax[k].set_yticks(np.arange(N_types), ticks)
    ax[0].set_title('Data')
    ax[1].set_title('Model')
    ax[0].imshow(rho_c_data,  cmap=cm.coolwarm, vmin=0.1, vmax=1.)
    ax[1].imshow(rho_c_model, cmap=cm.coolwarm, vmin=0.1, vmax=1.)
    for i in range(N_types):
        for j in range(N_types):
            ax[0].text(j, i, f'{rho_c_data[i, j]:.2f}',  ha='center', va='center', fontsize=12)
            ax[1].text(j, i, f'{rho_c_model[i, j]:.2f}', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150.)
    plt.close()
