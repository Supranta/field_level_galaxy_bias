"""fit_density.py — Parametric lognormal-Poisson fit over the full density field.

Fits a model where the mean galaxy count and lognormal scatter are explicit
parametric functions of the local matter density delta. All pixels are fit
simultaneously in a single MCMC run.

Config format uses nested fit_model / nested_model blocks:

    fit_model:
      mean_type:  neyrinck | powerlaw
      z_type:     shared   | zero
      sigma_type: density  | constant
      tidal_type: none     | s2
      num_warmup:  500
      num_samples: 500

    nested_model:             # optional; if absent, runs fit_model directly
      mean_type:  ...
      z_type:     ...
      sigma_type: ...
      tidal_type: none
      num_warmup: 100         # kept small — only 1 sample drawn

When nested_model is present, run_nuts_with_warmstart is used: a short nested
chain provides the initial parameter values for the fit chain.

Outputs
-------
<savedir>/samples.pkl          MCMC posterior samples.
<savedir>/log_likelihood.npy   Per-sample log-likelihood.

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

from lnp.data import load_data
from lnp.models import build_model
from lnp.inference import run_nuts, run_nuts_with_warmstart
from lnp.power_spectrum import Field2D


def _compute_s2_flat(delta_slab, box_size):
    """Compute the squared tidal field for each slab and return flattened."""
    N_grid = delta_slab.shape[-1]
    field  = Field2D(N_grid, box_size)
    s2_slabs = np.stack([field.get_s2_fields(delta_slab[i])
                         for i in range(delta_slab.shape[0])])
    return s2_slabs.flatten()


def _parse_model_block(block):
    """Extract build_model kwargs from a model config block."""
    return dict(
        mean_type  = block['mean_type'],
        z_type     = block['z_type'],
        sigma_type = block.get('sigma_type'),
        tidal_type = block.get('tidal_type', 'none'),
    )


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datafile   = cfg['datafile']
    savedir    = cfg['savedir']
    catalog    = cfg['catalog']
    fit_cfg    = cfg['fit_model']
    nested_cfg = cfg.get('nested_model')

    fit_kwargs  = _parse_model_block(fit_cfg)
    fit_warmup  = fit_cfg.get('num_warmup',  500)
    fit_samples = fit_cfg.get('num_samples', 500)

    delta_slab, Ng = load_data(datafile, catalog)
    N_types = Ng.shape[0]
    Ng_flat = Ng.reshape(N_types, -1)

    print("Fit model: mean=%s  z=%s  sigma=%s  tidal=%s"
          % (fit_kwargs['mean_type'], fit_kwargs['z_type'],
             fit_kwargs['sigma_type'], fit_kwargs['tidal_type']))
    if nested_cfg:
        nested_kwargs = _parse_model_block(nested_cfg)
        print("Nested:    mean=%s  z=%s  sigma=%s  tidal=%s"
              % (nested_kwargs['mean_type'], nested_kwargs['z_type'],
                 nested_kwargs['sigma_type'], nested_kwargs['tidal_type']))
    print("N_types: %d,  N_pix: %d" % (N_types, Ng_flat.shape[1]))

    needs_s2 = fit_kwargs['tidal_type'] == 's2'

    s2_flat = None
    if needs_s2:
        box_size = cfg.get('box_size', 1000.)
        print("Computing tidal field s2 (box_size=%.1f)..." % box_size)
        s2_flat = _compute_s2_flat(delta_slab, box_size)

    fit_model  = build_model(**fit_kwargs)
    os.makedirs(savedir, exist_ok=True)

    print("Running NUTS...")
    if nested_cfg is None:
        samples, log_lik = run_nuts(
            fit_model, Ng_flat,
            delta=delta_slab.flatten(),
            s2=s2_flat,
            num_warmup=fit_warmup, num_samples=fit_samples,
            compute_log_lik=True,
        )
    else:
        nested_model  = build_model(**nested_kwargs)
        nested_warmup = nested_cfg.get('num_warmup', 100)
        samples, log_lik = run_nuts_with_warmstart(
            nested_model, fit_model,
            Ng_flat, delta_slab.flatten(),
            s2=s2_flat,
            nested_warmup=nested_warmup,
            num_warmup=fit_warmup, num_samples=fit_samples,
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
        print("Usage: python fit_density.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
