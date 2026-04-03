"""model_comparison.py — Compare two density-pipeline models by WAIC and
harmonic-mean Bayes factor.

Usage
-----
    python3 model_comparison.py config1.yaml config2.yaml

Both configs must point to the *same* datafile (asserted at startup).
Each savedir must contain a ``log_likelihood.npy`` written by fit_density.py,
with shape ``(n_samples, n_pixels)``.
"""

import sys
import numpy as np
import yaml
from scipy.special import logsumexp


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def inference_mode(cfg):
    """Return 'sample' or 'optimize' from the top-level config key."""
    return cfg.get('inference_mode', 'sample')


def model_label(cfg):
    """Human-readable label from config keys."""
    fit      = cfg.get('fit_model', {})
    mean     = fit.get('mean_type',     '?')
    z        = fit.get('z_type',        '?')
    sigma    = fit.get('sigma_type',    '?')
    tidal    = fit.get('tidal_type',    'none')
    smoothed = fit.get('smoothed_type', 'none')
    mode     = inference_mode(cfg)
    return 'mean=%s  z=%s  sigma=%s  tidal=%s  smoothed=%s  mode=%s' % (
        mean, z, sigma, tidal, smoothed, mode)


# ─────────────────────────────────────────────────────────────────────────────
# Model-comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_waic(log_like_pixels):
    """WAIC from per-pixel log-likelihood samples.

    Parameters
    ----------
    log_like_pixels : (n_samples, n_pixels) ndarray

    Returns
    -------
    waic, lppd, p_waic : scalars
        Lower WAIC = better predictive fit.
    """
    n_samples = log_like_pixels.shape[0]
    lppd_per_pixel = logsumexp(log_like_pixels, axis=0) - np.log(n_samples)
    lppd   = lppd_per_pixel.sum()
    p_waic = np.var(log_like_pixels, axis=0, ddof=1).sum()
    waic   = -2.0 * (lppd - p_waic)
    return waic, lppd, p_waic


def compute_log_evidence_harmonic_mean(log_like_pixels, truncation_fraction=0.9):
    """Harmonic-mean estimator of log marginal likelihood.

    Retains only the top ``truncation_fraction`` of samples (by total
    log-likelihood) to reduce the instability of the raw harmonic mean.

    Parameters
    ----------
    log_like_pixels     : (n_samples, n_pixels) ndarray
    truncation_fraction : float in (0, 1]

    Returns
    -------
    log_evidence : scalar
    """
    log_like_samples = log_like_pixels.sum(axis=1)

    if truncation_fraction < 1.0:
        threshold = np.quantile(log_like_samples, 1.0 - truncation_fraction)
        kept = log_like_samples[log_like_samples >= threshold]
    else:
        kept = log_like_samples

    log_evidence = np.log(len(kept)) - logsumexp(-kept)
    return log_evidence


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print('Usage: python3 model_comparison.py config1.yaml config2.yaml')
        sys.exit(1)

    cfg1 = load_config(sys.argv[1])
    cfg2 = load_config(sys.argv[2])

    # Sanity check: model comparison is only valid on identical data.
    assert cfg1['datafile'] == cfg2['datafile'], (
        'datafile mismatch:\n  config1: %s\n  config2: %s\n'
        'Model comparison requires both models to be fit on the same data.'
        % (cfg1['datafile'], cfg2['datafile'])
    )
    assert cfg1['catalog'] == cfg2['catalog'], (
        'catalog mismatch:\n  config1: %s\n  config2: %s\n'
        'Model comparison requires both models to be fit on the same catalog.'
        % (cfg1['catalog'], cfg2['catalog'])
    )

    label1 = model_label(cfg1)
    label2 = model_label(cfg2)

    log_like1 = np.load('%s/log_likelihood.npy' % cfg1['savedir'])
    log_like2 = np.load('%s/log_likelihood.npy' % cfg2['savedir'])

    assert log_like1.ndim == 2 and log_like2.ndim == 2, (
        'log_likelihood.npy must have shape (n_samples, n_pixels)'
    )
    assert log_like1.shape[1] == log_like2.shape[1], (
        'Pixel count mismatch: model 1 has %d pixels, model 2 has %d'
        % (log_like1.shape[1], log_like2.shape[1])
    )

    is_map1 = log_like1.shape[0] == 1
    is_map2 = log_like2.shape[0] == 1
    NA      = '           N/A'

    max_lkl1 = log_like1.sum(axis=1).max()
    max_lkl2 = log_like2.sum(axis=1).max()

    if not is_map1:
        waic1, lppd1, p1 = compute_waic(log_like1)
        log_ev1           = compute_log_evidence_harmonic_mean(log_like1)
        mean_lkl1         = log_like1.sum(axis=1).mean()
    if not is_map2:
        waic2, lppd2, p2 = compute_waic(log_like2)
        log_ev2           = compute_log_evidence_harmonic_mean(log_like2)
        mean_lkl2         = log_like2.sum(axis=1).mean()

    col = 65
    print('=' * col)
    print('MODEL COMPARISON')
    print('=' * col)
    print('  Model 1: %s' % label1)
    print('  Model 2: %s' % label2)
    print('  Data   : %s' % cfg1['datafile'])
    print('-' * col)
    print('  n_samples  : %d  /  %d' % (log_like1.shape[0], log_like2.shape[0]))
    print('  n_pixels   : %d' % log_like1.shape[1])
    if is_map1 or is_map2:
        print('  NOTE: Bayesian metrics (WAIC, log evidence, mean log-lik) require')
        print('        multiple samples and are unavailable for MAP (optimize) runs.')
    print('=' * col)
    print('  Metric                    Model 1      Model 2      Delta (2-1)')
    print('-' * col)

    def _fmt(v1, v2):
        s1 = '%12.2f' % v1 if v1 is not None else NA
        s2 = '%12.2f' % v2 if v2 is not None else NA
        sd = '%12.2f' % (v2 - v1) if (v1 is not None and v2 is not None) else NA
        return s1, s2, sd

    w1 = waic1     if not is_map1 else None
    w2 = waic2     if not is_map2 else None
    l1 = lppd1     if not is_map1 else None
    l2 = lppd2     if not is_map2 else None
    p1_ = p1       if not is_map1 else None
    p2_ = p2       if not is_map2 else None
    e1 = log_ev1   if not is_map1 else None
    e2 = log_ev2   if not is_map2 else None
    m1 = mean_lkl1 if not is_map1 else None
    m2 = mean_lkl2 if not is_map2 else None

    print('  WAIC                 %s %s %s' % _fmt(w1, w2))
    print('  LPPD                 %s %s %s' % _fmt(l1, l2))
    print('  p_WAIC               %s %s %s' % _fmt(p1_, p2_))
    print('  Log evidence (HM)    %s %s %s' % _fmt(e1, e2))
    print('  Mean log-likelihood  %s %s %s' % _fmt(m1, m2))
    print('  Max  log-likelihood  %s %s %s' % _fmt(max_lkl1, max_lkl2))
    print('=' * col)
    print()

    # Verdicts — only for metrics available for both models.
    delta_max_lkl = max_lkl2 - max_lkl1
    max_winner    = 'Model 1' if max_lkl1 > max_lkl2 else 'Model 2'
    print('  Max log-lik favours  : %s  (delta = %+.2f)' % (max_winner, delta_max_lkl))

    if not is_map1 and not is_map2:
        delta_waic   = w2 - w1
        delta_log_ev = e2 - e1
        waic_winner  = 'Model 1' if w1 < w2 else 'Model 2'
        ev_winner    = 'Model 1' if e1 > e2 else 'Model 2'
        print('  WAIC favours         : %s  (delta = %+.2f)' % (waic_winner, delta_waic))
        print('  Log evidence favours : %s  (delta = %+.2f)' % (ev_winner,   delta_log_ev))
        print()
        print('  Interpretation of |delta log evidence|:')
        print('    < 1  : not worth more than a bare mention')
        print('    1-3  : positive evidence')
        print('    3-5  : strong evidence')
        print('    > 5  : very strong evidence')
    print('=' * col)


if __name__ == '__main__':
    main()
