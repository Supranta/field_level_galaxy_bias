"""fit_binned.py — Non-parametric lognormal-Poisson fit in density bins.

For each density bin, fits a shared-z lognormal-Poisson model with
constant rate and sigma parameters (joint_lognormal_model). This gives
a non-parametric view of how galaxy bias and stochasticity vary with
local matter density.

Outputs
-------
<savedir>/<n>.pkl          Per-bin summary statistics (mean, var, rho_c, rate, sigma).
<savedir>/figs/pdf/<n>.png           Per-bin count PDF comparison.
<savedir>/figs/corrcoef/<n>.png      Per-bin cross-correlation matrix.

Usage
-----
    python scripts/fit_binned.py <config.yaml>

See configs/binned_example.yaml for the expected config format.
"""
import os
import sys
import pickle

import numpy as np
import yaml

from lnp.data import load_data, compute_delta_bins, bin_mask
from lnp.models import joint_lognormal_model
from lnp.inference import run_nuts, posterior_predictive_binned, compute_summaries
from lnp.plotting import plot_count_pdfs, plot_corrcoef_matrix


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_delta_bins = cfg['n_delta_bins']
    datafile     = cfg['datafile']
    savedir      = cfg['savedir']
    catalog      = cfg['catalog']
    num_warmup   = cfg.get('num_warmup',  500)
    num_samples  = cfg.get('num_samples', 500)

    delta_slab, Ng = load_data(datafile, catalog)
    N_types    = Ng.shape[0]
    delta_flat = delta_slab.flatten()
    Ng_flat    = Ng.reshape(N_types, -1)
    delta_bins = compute_delta_bins(delta_flat, n_delta_bins)

    print("N_types: %d,  N_pix: %d" % (N_types, Ng_flat.shape[1]))

    os.makedirs(savedir,                      exist_ok=True)
    os.makedirs(savedir + '/figs/pdf',        exist_ok=True)
    os.makedirs(savedir + '/figs/corrcoef',   exist_ok=True)

    for n in range(n_delta_bins):
        print("=" * 60)
        print("Bin %d / %d" % (n + 1, n_delta_bins))
        print("=" * 60)

        mask    = bin_mask(delta_flat, delta_bins, n)
        Ng_bin  = Ng_flat[:, mask]

        samples = run_nuts(joint_lognormal_model, Ng_bin,
                           num_warmup=num_warmup, num_samples=num_samples)

        Ng_pp     = posterior_predictive_binned(samples, N_types)
        summaries = compute_summaries(Ng_pp, Ng_bin, samples)

        with open(savedir + '/%d.pkl' % n, 'wb') as f:
            pickle.dump(summaries, f)

        plot_count_pdfs(Ng_bin, Ng_pp, N_types,
                        savedir + '/figs/pdf/%d.png' % n)
        plot_corrcoef_matrix(summaries['data_rho_c'], summaries['model_rho_c'],
                             N_types, savedir + '/figs/corrcoef/%d.png' % n)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/fit_binned.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
