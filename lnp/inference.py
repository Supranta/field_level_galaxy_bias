import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, log_likelihood, init_to_value


def run_nuts(model, counts, delta=None, s2=None,
             num_warmup=500, num_samples=500,
             seed=42, compute_log_lik=False):
    """Run NUTS on a NumPyro model.

    Parameters
    ----------
    model : callable
        NumPyro model. Called as model(counts) if delta is None,
        else model(counts, delta), or model(counts, delta, s2) if s2 is given.
    counts : (N_types, N_pix) array-like of int
    delta : (N_pix,) array-like of float, optional
        Matter overdensity field. Required for the density model.
    s2 : (N_pix,) array-like of float, optional
        Squared tidal field. Required when tidal_type='s2'.
    num_warmup : int
    num_samples : int
    seed : int
    compute_log_lik : bool
        If True, also compute the per-sample log-likelihood summed over pixels
        and return it alongside the samples.

    Returns
    -------
    samples : dict
        MCMC posterior samples keyed by parameter name.
    log_lik : (n_samples,) ndarray
        Only returned when compute_log_lik=True.
    """
    counts_jax   = jnp.array(counts, dtype=jnp.int32)
    model_kwargs = {'counts': counts_jax}
    if delta is not None:
        model_kwargs['delta'] = jnp.array(delta, dtype=jnp.float32)
    if s2 is not None:
        model_kwargs['s2'] = jnp.array(s2, dtype=jnp.float32)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup,
                num_samples=num_samples, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs)
    samples = mcmc.get_samples()

    if not compute_log_lik:
        return samples

    log_liks = log_likelihood(model, samples, **model_kwargs)
    # log_lik  = np.array(log_liks['obs'].sum(-1))   # (n_samples,)
    log_lik  = np.array(log_liks['obs'])   # (n_samples,)
    return samples, log_lik


def run_nuts_with_warmstart(
    nested_model, fit_model,
    counts, delta,
    s2=None,
    nested_warmup=100,
    num_warmup=500, num_samples=500,
    seed=42, compute_log_lik=False,
):
    """Run NUTS with warm-start from a simpler nested model.

    Runs a short nested chain (nested_warmup warmup steps, 1 sample), then
    initialises the fit chain from that sample via init_to_value. Parameters
    absent from the nested model fall back to NumPyro's default init strategy.

    The nested model is always called without s2; s2 is passed only to the
    fit model, consistent with the nested model being a simpler variant.

    Parameters
    ----------
    nested_model : callable
        Simpler NumPyro model. Must accept (counts, delta).
    fit_model : callable
        Full model to sample from. Must accept (counts, delta), or
        (counts, delta, s2) when s2 is given.
    counts : (N_types, N_pix) int array-like
    delta : (N_pix,) float array-like
    s2 : (N_pix,) float array-like, optional
        Squared tidal field. Passed to fit_model only when not None.
    nested_warmup : int
        Warmup steps for the short nested chain.
    num_warmup, num_samples : int
    seed : int
    compute_log_lik : bool

    Returns
    -------
    samples : dict
    log_lik : (n_samples,) ndarray — only if compute_log_lik=True.
    """    
    counts_jax  = jnp.array(counts, dtype=jnp.int32)
    delta_jax   = jnp.array(delta, dtype=jnp.float32)
    nested_kwargs = {'counts': counts_jax,
                     'delta': delta_jax}
    fit_kwargs    = {'counts': counts_jax,
                     'delta': delta_jax}
    if s2 is not None:
        fit_kwargs['s2'] = jnp.array(s2, dtype=jnp.float32)

    # ── 1. Short nested chain ─────────────────────────────────────────────────
    print("Running nested chain (%d warmup, 1 sample)..." % nested_warmup)
    nested_mcmc = MCMC(NUTS(nested_model), num_warmup=nested_warmup,
                       num_samples=1, progress_bar=True)
    nested_mcmc.run(jax.random.PRNGKey(seed), **nested_kwargs)
    nested_sample = {k: v[0] for k, v in nested_mcmc.get_samples().items()}

    # ── 2. Run fit chain initialised from nested sample ───────────────────────
    print("Running fit chain (%d warmup, %d samples)..." % (num_warmup, num_samples))
    fit_nuts = NUTS(fit_model, init_strategy=init_to_value(values=nested_sample))
    fit_mcmc = MCMC(fit_nuts, num_warmup=num_warmup,
                    num_samples=num_samples, progress_bar=True)
    fit_mcmc.run(jax.random.PRNGKey(seed), **fit_kwargs)

    samples = fit_mcmc.get_samples()
    if not compute_log_lik:
        return samples

    log_liks = log_likelihood(fit_model, samples, **fit_kwargs)
    log_lik  = np.array(log_liks['obs'])
    return samples, log_lik


def posterior_predictive_binned(samples, N_types):
    """Draw posterior predictive galaxy counts from the joint_lognormal_model.

    Vectorises over all MCMC samples simultaneously, so the resulting
    array has N_pix * n_samples columns.

    Parameters
    ----------
    samples : dict
        MCMC samples with keys:
        - 'z'     : (n_samples, N_pix)
        - 'sigma' : (n_samples, N_types)
        - 'rate'  : (n_samples, N_types)
    N_types : int

    Returns
    -------
    Ng_pp : (N_types, n_samples * N_pix) ndarray of int
    """
    # shapes: z (S, P), sigma (S, T), rate (S, T)
    # broadcast to (S, P, T)
    LN_exp = (samples['z'][:, :, None] * samples['sigma'][:, None, :]
              - 0.5 * samples['sigma'][:, None, :] ** 2)
    lam   = np.exp(LN_exp)                            # (S, P, T)
    mu    = samples['rate'][:, None, :] * lam         # (S, P, T)
    Ng_pp = np.random.poisson(mu)                     # (S, P, T)
    return Ng_pp.reshape(-1, N_types).T               # (N_types, S*P)


def compute_summaries(Ng_pp, Ng_data, samples):
    """Compute mean, variance, and cross-correlation statistics.

    Parameters
    ----------
    Ng_pp : (N_types, N_pp) ndarray
        Posterior predictive galaxy counts.
    Ng_data : (N_types, N_pix) ndarray
        Observed galaxy counts.
    samples : dict
        Must contain at least 'rate' and 'sigma'.

    Returns
    -------
    dict with keys:
        model_mean, model_var, model_rho_c,
        data_mean,  data_var,  data_rho_c,
        rate, sigma
    """
    return {
        'model_mean':  Ng_pp.mean(1),
        'model_var':   Ng_pp.var(1),
        'model_rho_c': np.corrcoef(Ng_pp),
        'data_mean':   Ng_data.mean(1),
        'data_var':    Ng_data.var(1),
        'data_rho_c':  np.corrcoef(Ng_data),
        'rate':        np.array(samples['rate']),
        'sigma':       np.array(samples['sigma']),
    }
