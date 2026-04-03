import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from lnp.density_functions import neyrinck_model_jax, sigma_model_jax

# ---------------------------------------------------------------------------
# Binned model (non-parametric)
# ---------------------------------------------------------------------------

def joint_lognormal_model(counts):
    """Joint lognormal-Poisson model for a single density bin.

    All galaxy types share a common latent field z (one draw per pixel)
    but have independent per-type rate and lognormal scatter sigma.
    This model is fit independently for each density bin, so rate and
    sigma are non-parametric summaries of the bias at that density.

    Model
    -----
    sigma_t  ~ HalfNormal(1)
    rate_t   ~ Normal(mean(N_t), 10 * sqrt(mean(N_t) / N_pix))
    z_i      ~ Normal(0, 1)                         [shared across types]
    lam_i    = exp(sigma_t * z_i - sigma_t^2 / 2)  [mean=1 lognormal factor]
    N_t_i    ~ Poisson(rate_t * lam_i)

    Parameters
    ----------
    counts : (N_types, N_pix) int array
    """
    N_types, N_pix = counts.shape

    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0).expand([N_types]))
    rate  = numpyro.sample("rate",  dist.Normal(
        loc=counts.mean(axis=1),
        scale=10.0 * jnp.sqrt(counts.mean(axis=1) / N_pix),
    ))

    with numpyro.plate("pix", N_pix):
        z   = numpyro.sample("z", dist.Normal(0.0, 1.0))
        lam = jnp.exp(sigma[:, None] * z[None, :] - 0.5 * sigma[:, None] ** 2)
        mu  = rate[:, None] * lam
        numpyro.sample("obs",
                       dist.Independent(dist.Poisson(mu.T), 1),
                       obs=counts.T)


# ---------------------------------------------------------------------------
# Density model (parametric) — building blocks
# ---------------------------------------------------------------------------

def softened_uniform(name, low, high, shape=None, sigma=1.):
    """
    HMC-friendly replacement for numpyro Uniform(low, high).

    Samples an unconstrained Normal in R, then sigmoid-transforms
    into (low, high). Avoids hard boundary walls and Jacobian
    divergences that degrade HMC/NUTS performance.

    Parameters
    ----------
    name  : str
        NumPyro parameter name. The unconstrained sample is stored
        as f"{name}_raw"; the transformed value as `name`.
    low   : float
        Lower bound of the desired interval.
    high  : float
        Upper bound of the desired interval.
    shape : list/tuple or None
        Shape of the parameter array, e.g. [N_types].
        None for a scalar parameter.
    sigma : float
        Std dev of the Normal in unconstrained space (default 2.0).
        Larger -> flatter/more uniform in (low, high).
        Smaller -> more concentrated near the midpoint.

    Returns
    -------
    x : jnp array of shape `shape` (or scalar)
        Transformed parameter living in (low, high).
    """
    prior = dist.Normal(0.0, sigma)
    if shape is not None:
        prior = prior.expand(shape)

    x_raw = numpyro.sample(f"{name}_raw", prior)

    x = numpyro.deterministic(
        name,
        low + (high - low) * jax.nn.sigmoid(x_raw)
    )
    return x
    
def _sample_neyrinck_params(N_types):
    n_bar   = softened_uniform("n_bar",   1e-6,  100.0, shape=[N_types])
    beta    = softened_uniform("beta",    0.,   4.0,    shape=[N_types])
    delta_g = softened_uniform("delta_g", -0.7, 0.5,    shape=[N_types])
    return n_bar, beta, delta_g


def _sample_powerlaw_params(N_types):
    n_bar = softened_uniform("n_bar", 1e-6, 100.0, shape=[N_types])
    beta  = softened_uniform("beta",  0.,  4.0,    shape=[N_types])
    return n_bar, beta


def _sample_density_sigma_params(N_types):
    S       = softened_uniform("S",       0.05, 5.0,  shape=[N_types])
    gamma1  = softened_uniform("gamma1",  0.0,  1.0)
    gamma2  = softened_uniform("gamma2",  -5.0, 0.0)
    A_sigma = softened_uniform("A_sigma", 0.0,  0.5)
    return S, gamma1, gamma2, A_sigma


def _sample_constant_sigma_params(N_types):
    return softened_uniform("sigma", 0.0, 5.0, shape=[N_types])


def _neyrinck_mean(r, n_bar, beta, delta_g):
    """Evaluate Neyrinck mean.

    Parameters
    ----------
    r : (1, N_pix) or (N_types, N_pix) float array
        Effective density ratio, pre-broadcast by the caller.

    Returns
    -------
    (N_types, N_pix) float array
    """
    return neyrinck_model_jax(r, n_bar[:, None], beta[:, None], delta_g[:, None])


def _powerlaw_mean(r, n_bar, beta):
    """Evaluate power-law mean.

    Parameters
    ----------
    r : (1, N_pix) or (N_types, N_pix) float array
        Effective density ratio, pre-broadcast by the caller.

    Returns
    -------
    (N_types, N_pix) float array
    """
    return n_bar[:, None] * r ** beta[:, None]


def _density_sigma(r, S, gamma1, gamma2, A_sigma):
    """Evaluate density-dependent sigma, clipped to (1e-6, 4). Returns (N_types, N_pix).

    Parameters
    ----------
    r : (N_pix,) or (1, N_pix) or (N_types, N_pix) float array
        Density field. 1D input is expanded to (1, N_pix) for broadcasting.
    """
    r_2d = r[None, :] if r.ndim == 1 else r
    raw = sigma_model_jax(r_2d, S[:, None], gamma1, gamma2, A_sigma)
    return jnp.clip(raw, 1e-6, 4.0)


def _observe(rate, counts):
    numpyro.sample(
        "obs",
        dist.Independent(dist.Poisson(rate.T), 1),  # batch=(N_pix,), event=(N_types,)
        obs=counts.T,
    )


# ---------------------------------------------------------------------------
# Density model (parametric) — shared body
# ---------------------------------------------------------------------------

def _density_model_body(counts, r, mean_type, z_type, sigma_type,
                        tidal_type='none', s2=None,
                        smoothed_type='none', smooth_fields=None,
                        sigma_delta_type='plain'):
    """Shared model body for all density model variants.

    Parameters
    ----------
    counts : (N_types, N_pix) int array
    r : (N_pix,) float array
        Base density ratio 1 + delta (tidal correction applied here if needed).
    mean_type, z_type, sigma_type : str
        Same semantics as in build_model.
    tidal_type : {'none', 's2'}
        Whether to apply a tidal bias correction to the effective density.
    s2 : (N_pix,) float array or None
        Squared tidal field. Required when tidal_type == 's2'.
    smoothed_type : {'none', 'shared'}
        Whether to include smoothed-field bias terms in the effective density.
    smooth_fields : (n_scales, N_pix) float array or None
        Band-pass filtered fields. Required when smoothed_type != 'none'.
    sigma_delta_type : {'plain', 'effective'}
        Which density field sigma uses when sigma_type='density'.
        - 'plain'     : sigma uses clip(1 + delta, 1e-6)
        - 'effective' : sigma uses r_mean (after tidal + smoothed corrections)
        Ignored when sigma_type='constant'.
    """
    N_types, N_pix = counts.shape

    assert tidal_type in ('none', 's2')
    if tidal_type == 's2':
        assert s2 is not None, "s2 must be provided when tidal_type != 'none'"
    assert smoothed_type in ('none', 'shared')
    if smoothed_type != 'none':
        assert smooth_fields is not None, "smooth_fields required when smoothed_type != 'none'"
    assert sigma_delta_type in ('plain', 'effective')

    # r_sigma: plain density used by sigma (never shifted by bias terms)
    r_sigma = jnp.clip(r, a_min=1e-6)

    # r_mean: effective density for the mean, accumulates bias corrections
    r_mean = r[None, :]  # (1, N_pix)

    if tidal_type == 's2':
        b_s2   = numpyro.sample("b_s2", dist.Normal(0.0, 2.0))
        r_mean = r_mean + b_s2 * s2[None, :]

    if smoothed_type == 'shared':
        n_scales = smooth_fields.shape[0]
        b_smooth = numpyro.sample("b_smooth", dist.Normal(0.0, 2.0).expand([n_scales]))
        # dot over scales: b_smooth (n_scales,) @ smooth_fields (n_scales, N_pix) -> (N_pix,)
        r_mean = r_mean + jnp.dot(b_smooth, smooth_fields)[None, :]

    r_mean = jnp.clip(r_mean, a_min=1e-6)

    # ---- Mean ----
    if mean_type == 'neyrinck':
        n_bar, beta, delta_g = _sample_neyrinck_params(N_types)
        mu_det = _neyrinck_mean(r_mean, n_bar, beta, delta_g)
    else:  # 'powerlaw'
        n_bar, beta = _sample_powerlaw_params(N_types)
        mu_det = _powerlaw_mean(r_mean, n_bar, beta)

    # ---- Pure Poisson (no lognormal scatter) ----
    if z_type == 'zero':
        with numpyro.plate("pix", N_pix):
            _observe(mu_det, counts)
        return

    # ---- Sigma ----
    if sigma_type == 'density':
        S, gamma1, gamma2, A_sigma = _sample_density_sigma_params(N_types)
        r_for_sigma = r_mean if sigma_delta_type == 'effective' else r_sigma
    else:
        sigma_t = _sample_constant_sigma_params(N_types)

    # ---- Shared z: one draw per pixel, shared across all types ----
    with numpyro.plate("pix", N_pix):
        z     = numpyro.sample("z", dist.Normal(0.0, 1.0))
        sigma = (_density_sigma(r_for_sigma, S, gamma1, gamma2, A_sigma)
                 if sigma_type == 'density'
                 else jnp.clip(sigma_t[:, None] * jnp.ones((1, N_pix)), 1e-6, 4.0))
        lam   = jnp.exp(sigma * z[None, :] - 0.5 * sigma ** 2)
        _observe(mu_det * lam, counts)


# ---------------------------------------------------------------------------
# Density model (parametric) — factory
# ---------------------------------------------------------------------------

def build_model(mean_type, z_type, sigma_type=None, tidal_type='none', smoothed_type='none',
                sigma_delta_type='plain'):
    """Build a NumPyro model from six orthogonal design choices.

    Parameters
    ----------
    mean_type : {'neyrinck', 'powerlaw'}
        Functional form for the mean galaxy count as a function of density.
        - 'neyrinck'  : n_bar * r^beta * exp(-rho_g / r), per-type delta_g
        - 'powerlaw'  : n_bar * r^beta  (no void suppression)
    z_type : {'shared', 'zero'}
        How the lognormal latent field z is drawn per pixel.
        - 'shared' : one z per pixel, shared across all galaxy types
                     (induces cross-correlations between types)
        - 'zero'   : no lognormal scatter; reduces to pure Poisson
    sigma_type : {'density', 'constant'}, optional
        How the lognormal scatter sigma varies with density.
        Ignored when z_type='zero'.
        - 'density'  : sigma(r) = S * (r^gamma1 + A_sigma * r^gamma2)
        - 'constant' : sigma is a per-type constant
    tidal_type : {'none', 's2'}
        Whether to include a tidal bias term in the effective density r_mean.
        - 'none' : r_mean = 1 + delta
        - 's2'   : r_mean = 1 + delta + b_s2 * s2, scalar b_s2 ~ Normal(0, 2)
    smoothed_type : {'none', 'shared'}
        Whether to include smoothed-field bias terms in the effective density r_mean.
        Band-pass fields band_R = delta_smooth(R_{i+1}) - delta_smooth(R_i) are
        provided as smooth_fields at call time.
        - 'none'   : no smoothed bias
        - 'shared' : r_mean += sum_R b_smooth[R] * band_R, scalar b_smooth per scale
    sigma_delta_type : {'plain', 'effective'}
        Which density field sigma uses when sigma_type='density'.
        - 'plain'     : sigma uses clip(1 + delta, 1e-6)  (default)
        - 'effective' : sigma uses r_mean (after tidal + smoothed corrections);
                        when r_mean is (N_types, N_pix) each type gets its own sigma density
        Ignored when sigma_type='constant'.

    Returns
    -------
    model : callable
        NumPyro model with signature model(counts, delta, s2=None, smooth_fields=None).
    """
    if mean_type not in ('neyrinck', 'powerlaw'):
        raise ValueError("mean_type must be 'neyrinck' or 'powerlaw', got '%s'" % mean_type)
    if z_type not in ('shared', 'zero'):
        raise ValueError("z_type must be 'shared' or 'zero', got '%s'" % z_type)
    if z_type != 'zero' and sigma_type not in ('density', 'constant'):
        raise ValueError("sigma_type must be 'density' or 'constant', got '%s'" % sigma_type)
    if tidal_type not in ('none', 's2'):
        raise ValueError("tidal_type must be 'none' or 's2', got '%s'" % tidal_type)
    if smoothed_type not in ('none', 'shared'):
        raise ValueError("smoothed_type must be 'none' or 'shared', got '%s'" % smoothed_type)
    if sigma_delta_type not in ('plain', 'effective'):
        raise ValueError("sigma_delta_type must be 'plain' or 'effective', got '%s'" % sigma_delta_type)

    def model(counts, delta, s2=None, smooth_fields=None):
        _density_model_body(counts, 1.0 + delta, mean_type, z_type, sigma_type,
                            tidal_type=tidal_type, s2=s2,
                            smoothed_type=smoothed_type, smooth_fields=smooth_fields,
                            sigma_delta_type=sigma_delta_type)

    return model
