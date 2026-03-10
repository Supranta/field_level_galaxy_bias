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
    delta_slab : (N_slabs, H, W) ndarray
        Matter overdensity field (multiple 2D slabs).
    Ng : (N_types, N_slabs, H, W) ndarray
        Galaxy counts per type.
    """
    with h5.File(datafile, 'r') as f:
        delta_slab = f['delta_2d'][:]
        Ng         = f[catalog]['Ng'][:]
    return delta_slab, Ng


def compute_smoothed_fields(delta, smoothing_scales, box_size):
    """Compute multi-scale Gaussian-smoothed density fields via FFT.

    Each 2D slab is smoothed independently. The unsmoothed field is always
    included as the first entry in the output.

    Parameters
    ----------
    delta : (N_slabs, H, W) ndarray
        Matter overdensity field (multiple 2D slabs).
    smoothing_scales : list of float
        Smoothing scales in the same physical units as box_size.
        A Gaussian kernel exp(-0.5 * l^2 * scale^2) is applied in Fourier space.
    box_size : float
        Physical side length of each 2D slab (same units as smoothing_scales).

    Returns
    -------
    delta_fields : (N_scales + 1, N_slabs, H, W) ndarray
        Index 0 : unsmoothed delta.
        Indices 1..N : delta smoothed at each scale in smoothing_scales.
    """
    N_slabs, H, W = delta.shape
    N_Y = W // 2 + 1

    kx = 2.0 * np.pi * np.fft.fftfreq(H, d=box_size / H)
    ky = 2.0 * np.pi * np.fft.rfftfreq(W, d=box_size / W)
    kx_grid = np.tile(kx[:, None], (1, N_Y))
    ky_grid = np.tile(ky[None, :], (H, 1))
    k2_grid = kx_grid ** 2 + ky_grid ** 2     # (H, N_Y), no epsilon needed in filter

    smoothed_slabs = [delta]                   # index 0: unsmoothed
    for scale in smoothing_scales:
        filt = np.exp(-0.5 * k2_grid * scale ** 2)   # (H, N_Y)
        smoothed = np.stack([
            np.fft.irfft2(filt * np.fft.rfft2(delta[i]))
            for i in range(N_slabs)
        ])                                            # (N_slabs, H, W)
        smoothed_slabs.append(smoothed)
    smoothed_slabs = np.array(smoothed_slabs)
    # smoothed_slabs = smoothed_slabs[1:] - smoothed_slabs[:-1]
    # smoothed_slabs = np.vstack([delta[np.newaxis], smoothed_slabs])
    return smoothed_slabs

def compute_tidal_field(delta_2d, box_size):
    """Compute the squared tidal field s^2 = s_ij s_ij for each 2D slab.

    The tidal tensor components are computed in Fourier space as:
        s_ij(k) = (k_i k_j / k^2) * delta(k)

    and the scalar s^2 = s_xx^2 + s_yy^2 + 2 s_xy^2.
    The k=0 mode is set to zero (the tidal field has no DC contribution).

    Parameters
    ----------
    delta_2d : (N_slabs, H, W) ndarray
        Matter overdensity field (multiple 2D slabs).
    box_size : float
        Physical side length of each 2D slab.

    Returns
    -------
    s2 : (N_slabs, H, W) ndarray
        Squared tidal field per slab.
    """
    N_slabs, H, W = delta_2d.shape
    N_Y = W // 2 + 1

    kx = 2.0 * np.pi * np.fft.fftfreq(H, d=box_size / H)
    ky = 2.0 * np.pi * np.fft.rfftfreq(W, d=box_size / W)
    kx_grid = np.tile(kx[:, None], (1, N_Y))
    ky_grid = np.tile(ky[None, :], (H, 1))
    k2_grid = kx_grid ** 2 + ky_grid ** 2

    # Avoid division by zero at k=0; tidal field carries no DC component
    k2_safe = np.where(k2_grid == 0, 1.0, k2_grid)

    s2_slabs = []
    for i in range(N_slabs):
        delta_fft = np.fft.rfft2(delta_2d[i])
        s_xx = np.fft.irfft2(delta_fft * (kx_grid ** 2 / k2_safe), s=(H, W))
        s_yy = np.fft.irfft2(delta_fft * (ky_grid ** 2 / k2_safe), s=(H, W))
        s_xy = np.fft.irfft2(delta_fft * (kx_grid * ky_grid / k2_safe), s=(H, W))
        s2_slabs.append(s_xx ** 2 + s_yy ** 2 + 2.0 * s_xy ** 2)

    return np.stack(s2_slabs)   # (N_slabs, H, W)
    
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
