"""generate_synthetic.py — Generate a synthetic galaxy field from a fitted density model.

Reads posterior MCMC samples produced by fit_density.py and generates a
single synthetic realisation of the galaxy counts by:

  1. Picking one random posterior sample for all global parameters
     (everything except z — the per-pixel latent field).
  2. Drawing a fresh z ~ Normal(0, 1) per pixel, independent of the fit.
  3. Running the forward model to produce Poisson-sampled galaxy counts.

Results are written into the *same* HDF5 file as the input data, under a
new 'synthetic' catalog group, mirroring the structure of real catalogs:

    <datafile>
      synthetic/
        Ng         : (N_types, N_slabs, H, W)  — synthetic galaxy counts
        z          : (N_slabs, H, W)            — drawn latent field
        params/
          <name>   : scalar or (N_types,) array — chosen posterior parameters

Usage
-----
    python generate_synthetic.py <config.yaml>

The config file must be the same one used for fit_density.py.
"""
import os
import sys
import pickle

import h5py as h5
import numpy as np
import yaml

from lnp.data import load_data
from lnp.power_spectrum import Field2D


# ---------------------------------------------------------------------------
# Forward model (pure NumPy — no NumPyro needed for generation)
# ---------------------------------------------------------------------------

def _neyrinck_mean(r_mean, n_bar, beta, delta_g):
    """Evaluate the Neyrinck mean model.

    Parameters
    ----------
    r_mean : (N_types, N_pix) or (1, N_pix) array
    n_bar, beta, delta_g : (N_types,) arrays

    Returns
    -------
    (N_types, N_pix) array
    """
    rho_g    = (1.0 + delta_g)[:, None]                    # (N_types, 1)
    void_exp = np.maximum(-rho_g / r_mean, -10.0)
    return n_bar[:, None] * r_mean ** beta[:, None] * np.exp(void_exp)


def _powerlaw_mean(r_mean, n_bar, beta):
    """Evaluate the power-law mean model.

    Parameters
    ----------
    r_mean : (N_types, N_pix) or (1, N_pix) array
    n_bar, beta : (N_types,) arrays

    Returns
    -------
    (N_types, N_pix) array
    """
    return n_bar[:, None] * r_mean ** beta[:, None]


def _density_sigma(r, S, gamma1, gamma2, A_sigma):
    """Evaluate density-dependent lognormal scatter sigma(r).

    sigma(r) = S * (r^gamma1 + A_sigma * r^gamma2), clipped to (1e-6, 4).

    Parameters
    ----------
    r : (N_pix,) array — unshifted density ratio 1 + delta
    S : (N_types,) array
    gamma1, gamma2, A_sigma : scalars

    Returns
    -------
    (N_types, N_pix) array
    """
    raw = S[:, None] * (r[None, :] ** gamma1 + A_sigma * r[None, :] ** gamma2)
    return np.clip(raw, 1e-6, 4.0)


def forward_model(params, delta_flat, s2_flat,
                  mean_type, z_type, sigma_type, tidal_type, rng):
    """Generate synthetic galaxy counts from sampled global parameters.

    Replicates the probabilistic structure of _density_model_body, but
    draws z from its prior Normal(0,1) rather than conditioning on data.

    Parameters
    ----------
    params : dict
        Global model parameters extracted from a single posterior sample.
        Keys depend on mean_type / sigma_type / tidal_type.
    delta_flat : (N_pix,) array — flattened matter overdensity
    s2_flat : (N_pix,) array or None — flattened squared tidal field
    mean_type, z_type, sigma_type, tidal_type : str
        Same semantics as in build_model / _density_model_body.
    rng : numpy.random.Generator

    Returns
    -------
    Ng : (N_types, N_pix) int array — Poisson-sampled galaxy counts
    z  : (N_pix,) float array or None — drawn latent field (None if z_type='zero')
    """
    r = 1.0 + delta_flat   # (N_pix,)

    # ---- Tidal correction (mirrors _density_model_body) ----
    if tidal_type == 's2':
        assert s2_flat is not None
        b_s2 = float(params['b_s2'])
        r    = np.clip(r + b_s2 * s2_flat, 1e-6, None)   # shifted r used for mean AND sigma
        r_mean = r[None, :]                                # (1, N_pix)
    else:
        r      = np.clip(r, 1e-6, None)
        r_mean = r[None, :]                                # (1, N_pix)

    # ---- Mean (N_types, N_pix) ----
    if mean_type == 'neyrinck':
        mu_det = _neyrinck_mean(
            r_mean,
            np.asarray(params['n_bar']),
            np.asarray(params['beta']),
            np.asarray(params['delta_g']),
        )
    else:  # powerlaw
        mu_det = _powerlaw_mean(
            r_mean,
            np.asarray(params['n_bar']),
            np.asarray(params['beta']),
        )

    # ---- Pure Poisson (no lognormal scatter) ----
    if z_type == 'zero':
        Ng = rng.poisson(np.maximum(mu_det, 0.0)).astype(np.int32)
        return Ng, None

    # ---- Sigma (N_types, N_pix) ----
    # sigma always uses the unshifted (or s2-shifted-for-s2-mode) base r
    if sigma_type == 'density':
        sigma = _density_sigma(
            r,
            np.asarray(params['S']),
            float(params['gamma1']),
            float(params['gamma2']),
            float(params['A_sigma']),
        )
    else:  # constant
        N_pix = delta_flat.shape[0]
        sigma = np.clip(
            np.asarray(params['sigma'])[:, None] * np.ones((1, N_pix)), 1e-6, 4.0
        )

    # ---- Draw z from prior ----
    N_pix = delta_flat.shape[0]
    z = rng.standard_normal(N_pix).astype(np.float32)     # (N_pix,)

    lam = np.exp(sigma * z[None, :] - 0.5 * sigma ** 2)   # (N_types, N_pix)
    mu  = np.maximum(mu_det * lam, 0.0)
    Ng  = rng.poisson(mu).astype(np.int32)                 # (N_types, N_pix)

    return Ng, z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datafile  = cfg['datafile']
    savedir   = cfg['savedir']
    catalog   = cfg['catalog']
    fit_cfg   = cfg['fit_model']

    mean_type  = fit_cfg['mean_type']
    z_type     = fit_cfg['z_type']
    sigma_type = fit_cfg.get('sigma_type')
    tidal_type = fit_cfg.get('tidal_type', 'none')

    assert mean_type  in ('neyrinck', 'powerlaw'),        "Unknown mean_type: %s"  % mean_type
    assert z_type     in ('shared', 'zero'),              "Unknown z_type: %s"     % z_type
    assert tidal_type in ('none', 's2'),   "Unknown tidal_type: %s" % tidal_type
    if z_type != 'zero':
        assert sigma_type in ('density', 'constant'),     "Unknown sigma_type: %s" % sigma_type

    # ---- Load data ----
    delta_slab, Ng_real = load_data(datafile, catalog)
    slab_shape = delta_slab.shape          # (N_slabs, H, W)
    N_types    = Ng_real.shape[0]
    N_pix      = delta_slab.size
    delta_flat = delta_slab.flatten()

    print("Loaded delta field: shape=%s,  N_pix=%d" % (str(slab_shape), N_pix))
    print("N_types: %d" % N_types)

    # ---- Tidal field ----
    s2_flat = None
    if tidal_type == 's2':
        box_size = cfg.get('box_size', 1000.)
        print("Computing tidal field s2 (box_size=%.1f)..." % box_size)
        N_grid  = slab_shape[-1]
        field   = Field2D(N_grid, box_size)
        s2_flat = np.stack([
            field.get_s2_fields(delta_slab[i]) for i in range(slab_shape[0])
        ]).flatten()

    # ---- Load posterior samples ----
    samples_path = os.path.join(savedir, 'samples.pkl')
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)

    # Global keys: exclude per-pixel z and the unconstrained _raw counterparts.
    global_keys = [k for k in samples if k != 'z' and not k.endswith('_raw')]
    n_samples   = int(np.asarray(samples[global_keys[0]]).shape[0])
    print("Loaded %d posterior samples from %s" % (n_samples, samples_path))
    print("Global parameter keys: %s" % global_keys)

    # ---- Pick a random sample ----
    rng        = np.random.default_rng()
    sample_idx = int(rng.integers(0, n_samples))
    print("Using posterior sample index: %d" % sample_idx)

    params = {k: np.asarray(samples[k][sample_idx]) for k in global_keys}

    # ---- Generate synthetic counts ----
    print("Generating synthetic counts (mean_type=%s, z_type=%s, "
          "sigma_type=%s, tidal_type=%s)..."
          % (mean_type, z_type, sigma_type, tidal_type))

    Ng_synth, z_synth = forward_model(
        params, delta_flat, s2_flat,
        mean_type, z_type, sigma_type, tidal_type, rng,
    )

    # Reshape back to (N_types, N_slabs, H, W)
    Ng_synth_vol = Ng_synth.reshape(N_types, *slab_shape)

    # ---- Save to HDF5 ----
    print("Writing synthetic catalog to '%s' in %s..." % ('synthetic', datafile))
    with h5.File(datafile, 'a') as f:
        if 'synthetic' in f:
            del f['synthetic']
            print("Overwriting existing 'synthetic' group.")
        grp = f.create_group('synthetic')

        # Galaxy counts and latent field
        grp.create_dataset('Ng', data=Ng_synth_vol)
        if z_synth is not None:
            grp.create_dataset('z', data=z_synth.reshape(slab_shape))

        # Chosen parameter values — saved as individual datasets under params/
        params_grp = grp.create_group('params')
        for key, val in params.items():
            params_grp.create_dataset(key, data=val)

        # Provenance metadata
        grp.attrs['source_catalog'] = catalog
        grp.attrs['sample_index']   = sample_idx
        grp.attrs['mean_type']      = mean_type
        grp.attrs['z_type']         = z_type
        grp.attrs['sigma_type']     = sigma_type or 'none'
        grp.attrs['tidal_type']     = tidal_type

    print("Done.")
    print("  synthetic/Ng          : %s" % str(Ng_synth_vol.shape))
    if z_synth is not None:
        print("  synthetic/z           : %s" % str(slab_shape))
    print("  synthetic/params keys : %s" % list(params.keys()))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_synthetic.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
