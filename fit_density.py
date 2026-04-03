"""fit_density.py — Parametric lognormal-Poisson fit over the full density field.

Fits a model where the mean galaxy count and lognormal scatter are explicit
parametric functions of the local matter density delta. All pixels are fit
simultaneously in a single MCMC or MAP run.

Config format
-------------
Top-level keys:

    inference_mode: sample | optimize   # default: sample

    fit_model:
      mean_type:  neyrinck | powerlaw
      z_type:     shared   | zero
      sigma_type: density  | constant
      tidal_type: none     | s2

      # sampling (inference_mode: sample)
      num_warmup:  500
      num_samples: 500

      # optimisation (inference_mode: optimize)
      num_steps:    5000
      learning_rate: 0.01

    nested_model:             # optional; warms up the fit from a simpler model
      mean_type:  ...
      z_type:     ...
      sigma_type: ...
      tidal_type: none

      # sampling
      num_warmup: 100         # kept small — only 1 sample drawn

      # optimisation
      num_steps:    2000
      learning_rate: 0.01

When nested_model is present:
  - sample mode  : run_nuts_with_warmstart — short nested MCMC chain seeds the fit chain.
  - optimize mode: run_map_with_warmstart  — nested MAP result seeds the fit optimisation.

Outputs
-------
<savedir>/samples.pkl          Posterior samples (sample mode) or MAP point
                               estimate with a size-1 leading batch dimension
                               (optimize mode).
<savedir>/log_likelihood.npy   Shape (n_samples, n_pixels) for sample mode,
                               (1, n_pixels) for optimize mode.

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
from lnp.inference import (run_nuts, run_nuts_with_warmstart,
                           run_map, run_map_with_warmstart)
from lnp.power_spectrum import Field2D, compute_band_pass_fields, compute_s2_flat


def _parse_model_block(block):
    """Extract build_model kwargs from a model config block."""
    return dict(
        mean_type        = block['mean_type'],
        z_type           = block['z_type'],
        sigma_type       = block.get('sigma_type'),
        tidal_type       = block.get('tidal_type', 'none'),
        smoothed_type    = block.get('smoothed_type', 'none'),
        sigma_delta_type = block.get('sigma_delta_type', 'plain'),
    )


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    datafile   = cfg['datafile']
    savedir    = cfg['savedir']
    catalog    = cfg['catalog']
    fit_cfg    = cfg['fit_model']
    nested_cfg = cfg.get('nested_model')
    mode       = cfg.get('inference_mode', 'sample')

    assert mode in ('sample', 'optimize'), (
        "inference_mode must be 'sample' or 'optimize', got '%s'" % mode)

    fit_kwargs = _parse_model_block(fit_cfg)

    delta_slab, Ng = load_data(datafile, catalog)
    N_types = Ng.shape[0]
    Ng_flat = Ng.reshape(N_types, -1)

    print("inference_mode: %s" % mode)
    print("Fit model: mean=%s  z=%s  sigma=%s  tidal=%s  smoothed=%s  sigma_delta=%s"
          % (fit_kwargs['mean_type'], fit_kwargs['z_type'],
             fit_kwargs['sigma_type'], fit_kwargs['tidal_type'],
             fit_kwargs['smoothed_type'], fit_kwargs['sigma_delta_type']))
    if nested_cfg:
        nested_kwargs = _parse_model_block(nested_cfg)
        print("Nested:    mean=%s  z=%s  sigma=%s  tidal=%s  smoothed=%s  sigma_delta=%s"
              % (nested_kwargs['mean_type'], nested_kwargs['z_type'],
                 nested_kwargs['sigma_type'], nested_kwargs['tidal_type'],
                 nested_kwargs['smoothed_type'], nested_kwargs['sigma_delta_type']))
    print("N_types: %d,  N_pix: %d" % (N_types, Ng_flat.shape[1]))

    needs_s2         = fit_kwargs['tidal_type'] == 's2'
    needs_smooth     = fit_kwargs['smoothed_type'] != 'none'
    s2_flat          = None
    smooth_fields    = None
    smoothing_scales = []

    if needs_s2:
        box_size = cfg.get('box_size', 1000.)
        print("Computing tidal field s2 (box_size=%.1f)..." % box_size)
        s2_flat = compute_s2_flat(delta_slab, box_size)

    if needs_smooth:
        smoothing_scales = fit_cfg.get('smoothing_scales')
        assert smoothing_scales, "smoothing_scales must be set when smoothed_type != 'none'"
        box_size = cfg.get('box_size', 1000.)
        print("Computing band-pass fields (scales=%s, box_size=%.1f)..." % (smoothing_scales, box_size))
        smooth_fields = compute_band_pass_fields(delta_slab, smoothing_scales, box_size)

    fit_model = build_model(**fit_kwargs)
    os.makedirs(savedir, exist_ok=True)
    n_scales            = len(smoothing_scales) if needs_smooth else 0
    b_smooth_init_shape = (n_scales,) if needs_smooth else None

    if mode == 'sample':
        fit_warmup  = fit_cfg.get('num_warmup',  500)
        fit_samples = fit_cfg.get('num_samples', 500)
        print("Running NUTS...")
        if nested_cfg is None:
            samples, log_lik = run_nuts(
                fit_model, Ng_flat,
                delta=delta_slab.flatten(),
                s2=s2_flat,
                smooth_fields=smooth_fields,
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
                smooth_fields=smooth_fields,
                nested_warmup=nested_warmup,
                num_warmup=fit_warmup, num_samples=fit_samples,
                compute_log_lik=True,
                b_smooth_init_shape=b_smooth_init_shape,
            )

    else:  # optimize
        fit_steps = fit_cfg.get('num_steps',     5000)
        fit_lr    = fit_cfg.get('learning_rate', 0.01)
        print("Running MAP optimisation...")
        if nested_cfg is None:
            samples, log_lik = run_map(
                fit_model, Ng_flat,
                delta=delta_slab.flatten(),
                s2=s2_flat,
                smooth_fields=smooth_fields,
                num_steps=fit_steps, learning_rate=fit_lr,
                compute_log_lik=True,
            )
        else:
            nested_model  = build_model(**nested_kwargs)
            nested_steps  = nested_cfg.get('num_steps',     2000)
            nested_lr     = nested_cfg.get('learning_rate', 0.01)
            samples, log_lik = run_map_with_warmstart(
                nested_model, fit_model,
                Ng_flat, delta_slab.flatten(),
                s2=s2_flat,
                smooth_fields=smooth_fields,
                nested_steps=nested_steps, nested_lr=nested_lr,
                num_steps=fit_steps, learning_rate=fit_lr,
                compute_log_lik=True,
                b_smooth_init_shape=b_smooth_init_shape,
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
