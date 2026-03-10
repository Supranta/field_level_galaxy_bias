import numpy as np
import matplotlib.pyplot as plt
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
