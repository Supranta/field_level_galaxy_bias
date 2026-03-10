"""fit_density.py — Parametric lognormal-Poisson fit over the full density field.

Fits a model where the mean galaxy count and lognormal scatter are explicit
parametric functions of the local matter density delta. All pixels are fit
simultaneously in a single MCMC run.

Model variants are controlled by three config keys:
    mean_type  : 'neyrinck' | 'powerlaw'
    z_type     : 'shared' | 'independent' | 'zero'
    sigma_type : 'density' | 'constant'

Outputs
-------
<savedir>/samples.pkl          MCMC posterior samples.
<savedir>/log_likelihood.npy   Per-sample log-likelihood summed over pixels.

Usage
-----
    python scripts/fit_density.py <config.yaml>

See configs/density_example.yaml for the expected config format.
"""
import os
import sys
import pickle

import numpy as np
import yaml

from lnp.data import load_data
from lnp.models import build_model
from lnp.inference import run_nuts


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datafile    = cfg['datafile']
    savedir     = cfg['savedir']
    catalog     = cfg['catalog']
    mean_type   = cfg['mean_type']
    z_type      = cfg['z_type']
    sigma_type  = cfg.get('sigma_type')
    num_warmup  = cfg.get('num_warmup',  500)
    num_samples = cfg.get('num_samples', 500)

    delta_slab, Ng = load_data(datafile, catalog)
    N_types    = Ng.shape[0]
    delta_flat = delta_slab.flatten()
    Ng_flat    = Ng.reshape(N_types, -1)

    print("Model:   mean=%s  z=%s  sigma=%s" % (mean_type, z_type, sigma_type))
    print("N_types: %d,  N_pix: %d" % (N_types, Ng_flat.shape[1]))

    model = build_model(mean_type, z_type, sigma_type)

    os.makedirs(savedir, exist_ok=True)

    print("Running NUTS...")
    samples, log_lik = run_nuts(
        model, Ng_flat, delta=delta_flat,
        num_warmup=num_warmup, num_samples=num_samples,
        compute_log_lik=True,
    )

    savepath = savedir + '/samples.pkl'
    with open(savepath, 'wb') as f:
        pickle.dump(samples, f)
    print("Samples saved to", savepath)

    savepath = savedir + '/log_likelihood.npy'
    np.save(savepath, log_lik)
    print("Log likelihood saved to", savepath)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/fit_density.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
