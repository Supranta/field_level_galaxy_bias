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
    delta_g = softened_uniform("delta_g", -0.99, 0.5,    shape=[N_types])
    return n_bar, beta, delta_g


def _sample_neyrinck_shared_params(N_types):
    n_bar   = softened_uniform("n_bar",   1e-6,  100.0, shape=[N_types])
    beta    = softened_uniform("beta",    0.,   4.0,    shape=[N_types])
    delta_g = softened_uniform("delta_g", -0.99, 0.5)    # scalar, shared
    return n_bar, beta, jnp.broadcast_to(delta_g, (N_types,))


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
    """Evaluate Neyrinck mean. Returns (N_types, N_pix)."""
    return neyrinck_model_jax(r[None, :], n_bar[:, None], beta[:, None], delta_g[:, None])


def _powerlaw_mean(r, n_bar, beta):
    """Evaluate power-law mean. Returns (N_types, N_pix)."""
    return n_bar[:, None] * r[None, :] ** beta[:, None]


def _density_sigma(r, S, gamma1, gamma2, A_sigma):
    """Evaluate density-dependent sigma, clipped to (1e-6, 4). Returns (N_types, N_pix)."""
    raw = sigma_model_jax(r[None, :], S[:, None], gamma1, gamma2, A_sigma)
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

def _density_model_body(counts, r, mean_type, z_type, sigma_type):
    """Shared model body for both single-scale and multiscale variants.

    Parameters
    ----------
    counts : (N_types, N_pix) int array
    r : (N_pix,) float array
        Effective density ratio r_eff = 1 + delta_eff.
    mean_type, z_type, sigma_type : str
        Same semantics as in build_model.
    """
    N_types, N_pix = counts.shape

    # ---- Mean ----
    if mean_type == 'neyrinck':
        n_bar, beta, delta_g = _sample_neyrinck_params(N_types)
        mu_det = _neyrinck_mean(r, n_bar, beta, delta_g)
    elif mean_type == 'neyrinck_shared':
        n_bar, beta, delta_g = _sample_neyrinck_shared_params(N_types)
        mu_det = _neyrinck_mean(r, n_bar, beta, delta_g)
    else:  # 'powerlaw'
        n_bar, beta = _sample_powerlaw_params(N_types)
        mu_det = _powerlaw_mean(r, n_bar, beta)

    # ---- Pure Poisson (no lognormal scatter) ----
    if z_type == 'zero':
        with numpyro.plate("pix", N_pix):
            _observe(mu_det, counts)
        return

    # ---- Sigma ----
    if sigma_type == 'density':
        S, gamma1, gamma2, A_sigma = _sample_density_sigma_params(N_types)
    else:
        sigma_t = _sample_constant_sigma_params(N_types)

    # ---- Shared z: one draw per pixel, shared across all types ----
    with numpyro.plate("pix", N_pix):
        z     = numpyro.sample("z", dist.Normal(0.0, 1.0))
        sigma = (_density_sigma(r, S, gamma1, gamma2, A_sigma)
                 if sigma_type == 'density'
                 else jnp.clip(sigma_t[:, None] * jnp.ones((1, N_pix)), 1e-6, 4.0))
        lam   = jnp.exp(sigma * z[None, :] - 0.5 * sigma ** 2)
        _observe(mu_det * lam, counts)


# ---------------------------------------------------------------------------
# Density model (parametric) — factory
# ---------------------------------------------------------------------------

def build_model(mean_type, z_type, sigma_type=None, multiscale=False, tidal=False):
    """Build a NumPyro model from three orthogonal design choices.

    Parameters
    ----------
    mean_type : {'neyrinck', 'neyrinck_shared', 'powerlaw'}
        Functional form for the mean galaxy count as a function of density.
        - 'neyrinck'        : n_bar * r^beta * exp(-rho_g / r), per-type delta_g
        - 'neyrinck_shared' : n_bar * r^beta * exp(-rho_g / r), single delta_g shared across types
        - 'powerlaw'        : n_bar * r^beta  (no void suppression)
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
    multiscale : bool, optional
        If True, the effective density is a weighted combination of
        multi-scale smoothed fields:

            delta_eff = A_0 * delta_0 + A_1 * delta_1 + ... + A_N * delta_N

        Model receives delta_fields (N_scales, N_pix) instead of delta.
        Priors:
            A_0    = 1 (deterministic, unsmoothed anchor)
            A_1..N ~ Uniform(-10.0, 10.0)
    tidal : bool, optional
        If True, adds a tidal bias correction to the effective density:

            delta_eff += b_s2 * s2

        where s2 is the squared tidal field. b_s2 is shared across galaxy
        types. Model receives an additional s2 (N_pix,) argument.
        Prior:
            b_s2 ~ Uniform(-5.0, 5.0)

        Can be combined with multiscale=True.

    Returns
    -------
    model : callable
        NumPyro model. Signature:
            model(counts, delta)                         [base]
            model(counts, delta_fields)                  [multiscale]
            model(counts, delta, s2)                     [tidal]
            model(counts, delta_fields, s2)              [multiscale + tidal]
    """
    if mean_type not in ('neyrinck', 'neyrinck_shared', 'powerlaw'):
        raise ValueError("mean_type must be 'neyrinck', 'neyrinck_shared', or 'powerlaw', got '%s'" % mean_type)
    if z_type not in ('shared', 'zero'):
        raise ValueError("z_type must be 'shared' or 'zero', got '%s'" % z_type)
    if z_type != 'zero' and sigma_type not in ('density', 'constant'):
        raise ValueError("sigma_type must be 'density' or 'constant', got '%s'" % sigma_type)

    def _multiscale_delta_eff(delta_fields):
        """Compute weighted delta_eff from multi-scale fields. Returns (N_pix,)."""
        N_scales = delta_fields.shape[0]
        A_0 = numpyro.deterministic("A_0", jnp.array([1.]))
        if N_scales > 1:
            A_smooth = numpyro.sample(
                "A_smooth", dist.Uniform(-10.0, 10.0).expand([N_scales - 1])
            )
            A = jnp.concatenate([A_0, A_smooth])
        else:
            A = A_0
        return jnp.tensordot(A, delta_fields, axes=[[0], [0]])

    def _tidal_correction(delta_eff, s2):
        """Add tidal bias correction to delta_eff. Returns (N_pix,)."""
        # b_s2 = numpyro.sample("b_s2", dist.Uniform(-5.0, 5.0))
        b_s2 = numpyro.deterministic("b_s2", jnp.array([1.]))
        return delta_eff + b_s2 * s2

    if multiscale and tidal:
        def model(counts, delta_fields, s2):
            delta_eff = _multiscale_delta_eff(delta_fields)
            delta_eff = _tidal_correction(delta_eff, s2)
            delta_eff = (delta_eff / delta_eff.std()) * delta_fields[0].std()
            _density_model_body(counts, 1.0 + delta_eff, mean_type, z_type, sigma_type)

    elif multiscale:
        def model(counts, delta_fields):
            delta_eff = _multiscale_delta_eff(delta_fields)
            delta_eff = (delta_eff / delta_eff.std()) * delta_fields[0].std()
            _density_model_body(counts, 1.0 + delta_eff, mean_type, z_type, sigma_type)

    elif tidal:
        def model(counts, delta, s2):
            delta_eff = _tidal_correction(delta, s2)
            delta_eff = (delta_eff / delta_eff.std()) * delta.std()
            _density_model_body(counts, 1.0 + delta_eff, mean_type, z_type, sigma_type)

    else:
        def model(counts, delta):
            _density_model_body(counts, 1.0 + delta, mean_type, z_type, sigma_type)

    return model
