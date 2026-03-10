import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, log_likelihood


def run_nuts(model, counts, delta=None, delta_fields=None, s2=None,
             num_warmup=500, num_samples=500,
             seed=42, compute_log_lik=False):
    """Run NUTS on a NumPyro model.

    Parameters
    ----------
    model : callable
        NumPyro model. Called as model(counts) if delta is None,
        else model(counts, delta) or model(counts, delta_fields).
    counts : (N_types, N_pix) array-like of int
    delta : (N_pix,) array-like of float, optional
        Matter overdensity field. Required for the single-scale density model.
    delta_fields : (N_scales, N_pix) array-like of float, optional
        Multi-scale density fields for the multiscale model. Index 0 must be
        the unsmoothed field. Mutually exclusive with delta.
    s2 : (N_pix,) array-like of float, optional
        Squared tidal field. Required when the model was built with tidal=True.
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
    if delta is not None and delta_fields is not None:
        raise ValueError("Provide either delta or delta_fields, not both.")

    counts_jax    = jnp.array(counts, dtype=jnp.int32)
    model_kwargs  = {'counts': counts_jax}
    if delta is not None:
        model_kwargs['delta'] = jnp.array(delta, dtype=jnp.float32)
    if delta_fields is not None:
        model_kwargs['delta_fields'] = jnp.array(delta_fields, dtype=jnp.float32)
    if s2 is not None:
        model_kwargs['s2'] = jnp.array(s2, dtype=jnp.float32)

    mcmc = MCMC(NUTS(model), num_warmup=num_warmup,
                num_samples=num_samples, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs)
    samples = mcmc.get_samples()

    if not compute_log_lik:
        return samples

    log_liks = log_likelihood(model, samples, **model_kwargs)
    log_lik  = np.array(log_liks['obs'].sum(-1))   # (n_samples,)
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
