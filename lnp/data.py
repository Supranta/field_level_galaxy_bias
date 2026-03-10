import h5py as h5
import numpy as np


def load_data(datafile, catalog):
    """Load the 2D density field and galaxy counts from an HDF5 file.

    Parameters
    ----------
    datafile : str
        Path to the HDF5 file. Must contain a 'delta_2d' dataset and a
        group named <catalog> with an 'Ng' dataset.
    catalog : str
        HDF5 group name for the galaxy catalog.

    Returns
    -------
    delta_slab : (H, W) ndarray
        2D matter overdensity field.
    Ng : (N_types, H, W) ndarray
        Galaxy counts per type.
    """
    with h5.File(datafile, 'r') as f:
        delta_slab = f['delta_2d'][:]
        Ng         = f[catalog]['Ng'][:]
    return delta_slab, Ng


def compute_delta_bins(delta_flat, n_bins):
    """Compute percentile-equal bin edges over the delta distribution.

    Parameters
    ----------
    delta_flat : (N_pix,) ndarray
    n_bins : int

    Returns
    -------
    bins : (n_bins + 1,) ndarray
    """
    return np.array([
        np.percentile(delta_flat, 100.0 / n_bins * i)
        for i in range(n_bins + 1)
    ])


def compute_delta_mean(delta_flat, delta_bins):
    """Mean delta value in each bin.

    Parameters
    ----------
    delta_flat : (N_pix,) ndarray
    delta_bins : (n_bins + 1,) ndarray

    Returns
    -------
    delta_mean : (n_bins,) ndarray
    """
    n_bins = len(delta_bins) - 1
    return np.array([
        delta_flat[(delta_flat > delta_bins[i]) & (delta_flat <= delta_bins[i + 1])].mean()
        for i in range(n_bins)
    ])


def bin_mask(delta_flat, delta_bins, n):
    """Boolean mask selecting pixels in bin n.

    Parameters
    ----------
    delta_flat : (N_pix,) ndarray
    delta_bins : (n_bins + 1,) ndarray
    n : int

    Returns
    -------
    mask : (N_pix,) bool ndarray
    """
    return (delta_flat > delta_bins[n]) & (delta_flat <= delta_bins[n + 1])
