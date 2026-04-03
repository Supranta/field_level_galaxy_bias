import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, log_likelihood, init_to_value
from numpyro.infer.autoguide import AutoDelta


def run_nuts(model, counts, delta=None, s2=None, smooth_fields=None,
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
    smooth_fields : (n_scales, N_pix) array-like of float, optional
        Band-pass filtered fields. Required when smoothed_type != 'none'.
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
    model_kwargs = _make_model_kwargs(counts, delta, s2, smooth_fields)

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
    smooth_fields=None,
    nested_warmup=100,
    num_warmup=500, num_samples=500,
    seed=42, compute_log_lik=False,
    b_smooth_init_shape=None,
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
    nested_kwargs = _make_model_kwargs(counts, delta)
    fit_kwargs    = _make_model_kwargs(counts, delta, s2, smooth_fields)

    # ── 1. Short nested chain ─────────────────────────────────────────────────
    print("Running nested chain (%d warmup, 1 sample)..." % nested_warmup)
    nested_mcmc = MCMC(NUTS(nested_model), num_warmup=nested_warmup,
                       num_samples=1, progress_bar=True)
    nested_mcmc.run(jax.random.PRNGKey(seed), **nested_kwargs)
    nested_sample = {k: v[0] for k, v in nested_mcmc.get_samples().items()}

    # ── 2. Run fit chain initialised from nested sample ───────────────────────
    if s2 is not None:
        nested_sample['b_s2'] = jnp.float32(0.0)
    if smooth_fields is not None and b_smooth_init_shape is not None:
        nested_sample['b_smooth'] = jnp.zeros(b_smooth_init_shape, dtype=jnp.float32)
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


# ---------------------------------------------------------------------------
# MAP optimisation via SVI + AutoDelta
# ---------------------------------------------------------------------------

def _make_model_kwargs(counts, delta=None, s2=None, smooth_fields=None):
    """Build the keyword dict passed to a model callable."""
    kwargs = {'counts': jnp.array(counts, dtype=jnp.int32)}
    if delta is not None:
        kwargs['delta'] = jnp.array(delta, dtype=jnp.float32)
    if s2 is not None:
        kwargs['s2'] = jnp.array(s2, dtype=jnp.float32)
    if smooth_fields is not None:
        kwargs['smooth_fields'] = jnp.array(smooth_fields, dtype=jnp.float32)
    return kwargs


def _svi_map(model, model_kwargs, num_steps, learning_rate, seed,
             init_loc_fn=None):
    """Run SVI with AutoDelta to find the MAP estimate.

    Parameters
    ----------
    model       : NumPyro model callable
    model_kwargs: dict of data arrays to pass to model
    num_steps   : int
    learning_rate: float
    seed        : int
    init_loc_fn : numpyro init strategy or None
        If given, passed to AutoDelta to warm-start the optimisation.

    Returns
    -------
    map_params : dict
        MAP point estimate for every latent site (no batch dimension).
    final_loss : float
        Final SVI loss (negative log joint at MAP).
    """
    guide_kwargs = {} if init_loc_fn is None else {'init_loc_fn': init_loc_fn}
    guide  = AutoDelta(model, **guide_kwargs)
    svi    = SVI(model, guide, optax.adam(learning_rate), loss=Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(seed), num_steps, **model_kwargs,
                     progress_bar=True)
    map_params  = guide.median(result.params)
    final_loss  = float(result.losses[-1])
    return map_params, final_loss


def run_map(model, counts, delta=None, s2=None, smooth_fields=None,
            num_steps=5000, learning_rate=0.01,
            seed=42, compute_log_lik=False):
    """Find the MAP estimate of a NumPyro model via SVI with an AutoDelta guide.

    Parameters
    ----------
    model        : NumPyro model callable
    counts       : (N_types, N_pix) int array-like
    delta        : (N_pix,) float array-like, optional
    s2           : (N_pix,) float array-like, optional
        Squared tidal field. Required when tidal_type != 'none'.
    num_steps    : int
        Number of SVI optimisation steps.
    learning_rate: float
    seed         : int
    compute_log_lik : bool

    Returns
    -------
    samples : dict
        MAP parameter values, each wrapped in a size-1 leading batch dimension
        for compatibility with downstream code that expects MCMC-style dicts.
    log_lik : (1, N_pix) ndarray
        Only returned when compute_log_lik=True.
    """
    model_kwargs        = _make_model_kwargs(counts, delta, s2, smooth_fields)
    map_params, loss    = _svi_map(model, model_kwargs, num_steps,
                                   learning_rate, seed)
    print("MAP final loss: %.4f" % loss)

    # Wrap in size-1 batch dim so log_likelihood and downstream scripts work.
    samples = {k: v[None] for k, v in map_params.items()}

    if not compute_log_lik:
        return samples

    log_liks = log_likelihood(model, samples, **model_kwargs)
    log_lik  = np.array(log_liks['obs'])   # (1, N_pix)
    return samples, log_lik


def run_map_with_warmstart(
    nested_model, fit_model,
    counts, delta,
    s2=None,
    smooth_fields=None,
    nested_steps=2000, nested_lr=0.01,
    num_steps=5000, learning_rate=0.01,
    seed=42, compute_log_lik=False,
    b_smooth_init_shape=None,
):
    """Find the MAP estimate with warm-start from a simpler nested model.

    Optimises the nested model first (without s2), then uses its MAP params
    to initialise the fit model optimisation.  Parameters absent from the
    nested model fall back to AutoDelta's default initialisation.

    Parameters
    ----------
    nested_model : NumPyro model callable — simpler model, called without s2.
    fit_model    : NumPyro model callable — full model.
    counts       : (N_types, N_pix) int array-like
    delta        : (N_pix,) float array-like
    s2           : (N_pix,) float array-like, optional
    nested_steps : int  — SVI steps for the nested model.
    nested_lr    : float — learning rate for the nested model.
    num_steps    : int  — SVI steps for the fit model.
    learning_rate: float — learning rate for the fit model.
    seed         : int
    compute_log_lik : bool

    Returns
    -------
    samples  : dict  (size-1 batch dimension)
    log_lik  : (1, N_pix) ndarray — only if compute_log_lik=True.
    """
    nested_kwargs = _make_model_kwargs(counts, delta)
    fit_kwargs    = _make_model_kwargs(counts, delta, s2, smooth_fields)

    # ── 1. Optimise nested model ───────────────────────────────────────────────
    print("Optimising nested model (%d steps)..." % nested_steps)
    nested_map, nested_loss = _svi_map(nested_model, nested_kwargs,
                                       nested_steps, nested_lr, seed)
    print("Nested MAP final loss: %.4f" % nested_loss)

    # Inject zero-initialised b_s2 for the tidal parameters the nested model
    # does not have.
    if s2 is not None:
        nested_map['b_s2'] = jnp.float32(0.0)
    if smooth_fields is not None and b_smooth_init_shape is not None:
        nested_map['b_smooth'] = jnp.zeros(b_smooth_init_shape, dtype=jnp.float32)

    # ── 2. Optimise fit model, warm-started from nested MAP ───────────────────
    print("Optimising fit model (%d steps)..." % num_steps)
    init_loc = init_to_value(values=nested_map)
    map_params, loss = _svi_map(fit_model, fit_kwargs, num_steps,
                                learning_rate, seed, init_loc_fn=init_loc)
    print("MAP final loss: %.4f" % loss)

    samples = {k: v[None] for k, v in map_params.items()}

    if not compute_log_lik:
        return samples

    log_liks = log_likelihood(fit_model, samples, **fit_kwargs)
    log_lik  = np.array(log_liks['obs'])   # (1, N_pix)
    return samples, log_lik
