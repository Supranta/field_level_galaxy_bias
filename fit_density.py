"""fit_density.py — Parametric lognormal-Poisson fit over the full density field.

Fits a model where the mean galaxy count and lognormal scatter are explicit
parametric functions of the local matter density delta. All pixels are fit
simultaneously in a single MCMC run.

Model variants are controlled by config keys:
    mean_type  : 'neyrinck' | 'powerlaw'
    z_type     : 'shared' | 'zero'
    sigma_type : 'density' | 'constant'

Multiscale mode (smoothing_scales in config):
    delta_eff = A_0 * delta_0 + A_1 * delta_1 + ... + A_N * delta_N

Tidal bias mode (tidal: true in config):
    delta_eff += b_s2 * s2

Both modes can be combined.

Outputs
-------
<savedir>/samples.pkl          MCMC posterior samples.
<savedir>/log_likelihood.npy   Per-sample log-likelihood summed over pixels.

Usage
-----
    python fit_density.py <config.yaml>

See configs/density_example.yaml for the expected config format.
"""
import os
import sys
import pickle

import numpy as np
import yaml

from lnp.data import load_data, compute_smoothed_fields, compute_tidal_field
from lnp.models import build_model
from lnp.inference import run_nuts


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datafile         = cfg['datafile']
    savedir          = cfg['savedir']
    catalog          = cfg['catalog']
    mean_type        = cfg['mean_type']
    z_type           = cfg['z_type']
    sigma_type       = cfg.get('sigma_type')
    smoothing_scales = cfg.get('smoothing_scales')   # None → single-scale mode
    box_size         = cfg.get('box_size')           # required for multiscale or tidal
    tidal            = cfg.get('tidal', False)
    num_warmup       = cfg.get('num_warmup',  500)
    num_samples      = cfg.get('num_samples', 500)

    multiscale = smoothing_scales is not None

    if (multiscale or tidal) and box_size is None:
        raise ValueError("box_size must be set in the config when using multiscale or tidal.")

    delta_slab, Ng = load_data(datafile, catalog)
    N_types = Ng.shape[0]
    Ng_flat = Ng.reshape(N_types, -1)

    print("Model:   mean=%s  z=%s  sigma=%s  multiscale=%s  tidal=%s"
          % (mean_type, z_type, sigma_type, multiscale, tidal))
    print("N_types: %d,  N_pix: %d" % (N_types, Ng_flat.shape[1]))

    model = build_model(mean_type, z_type, sigma_type,
                        multiscale=multiscale, tidal=tidal)

    os.makedirs(savedir, exist_ok=True)

    # ---- Prepare data fields ----
    nuts_kwargs = dict(num_warmup=num_warmup, num_samples=num_samples,
                       compute_log_lik=True)

    if multiscale:
        delta_fields_2d = compute_smoothed_fields(delta_slab, smoothing_scales, box_size)
        N_scales_total  = delta_fields_2d.shape[0]
        print("Smoothing scales: %s  (total fields: %d)" % (smoothing_scales, N_scales_total))
        nuts_kwargs['delta_fields'] = delta_fields_2d.reshape(N_scales_total, -1)
    else:
        nuts_kwargs['delta'] = delta_slab.flatten()

    if tidal:
        s2_flat = compute_tidal_field(delta_slab, box_size).flatten()
        print("delta_slab.std: "+str(delta_slab.std()))
        s2_flat = (s2_flat / s2_flat.std()) * delta_slab.std()
        print("s2.std: "+str(s2_flat.std()))
        nuts_kwargs['s2'] = s2_flat

    print("Running NUTS...")
    samples, log_lik = run_nuts(model, Ng_flat, **nuts_kwargs)

    savepath = savedir + '/samples.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(samples, f)
    print("Samples saved to", savepath)

    savepath = savedir + '/log_likelihood.npy'
    np.save(savepath, log_lik)
    print("Log likelihood saved to", savepath)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fit_density.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
