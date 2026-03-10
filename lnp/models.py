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

def _sample_neyrinck_params(N_types):
    n_bar   = numpyro.sample("n_bar",   dist.Uniform(1e-6, 100.0).expand([N_types]))
    beta    = numpyro.sample("beta",    dist.Uniform(0.0, 5.0).expand([N_types]))
    delta_g = numpyro.sample("delta_g", dist.Uniform(-0.99, -0.1).expand([N_types]))
    return n_bar, beta, delta_g


def _sample_powerlaw_params(N_types):
    n_bar = numpyro.sample("n_bar", dist.Uniform(1e-6, 100.0).expand([N_types]))
    beta  = numpyro.sample("beta",  dist.Uniform(0.0, 5.0).expand([N_types]))
    return n_bar, beta


def _sample_density_sigma_params(N_types):
    S       = numpyro.sample("S",       dist.Uniform(0.05, 5.0).expand([N_types]))
    gamma1  = numpyro.sample("gamma1",  dist.Uniform(0.0, 1.0))
    gamma2  = numpyro.sample("gamma2",  dist.Uniform(-5.0, 0.0))
    A_sigma = numpyro.sample("A_sigma", dist.Uniform(0.0, 0.1))
    return S, gamma1, gamma2, A_sigma


def _sample_constant_sigma_params(N_types):
    return numpyro.sample("sigma", dist.Uniform(0.0, 5.0).expand([N_types]))


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
# Density model (parametric) — factory
# ---------------------------------------------------------------------------

def build_model(mean_type, z_type, sigma_type=None):
    """Build a NumPyro model from three orthogonal design choices.

    Parameters
    ----------
    mean_type : {'neyrinck', 'powerlaw'}
        Functional form for the mean galaxy count as a function of density.
        - 'neyrinck' : n_bar * r^beta * exp(-rho_g / r)  (void suppression)
        - 'powerlaw' : n_bar * r^beta
    z_type : {'shared', 'independent', 'zero'}
        How the lognormal latent field z is drawn per pixel.
        - 'shared'      : one z per pixel, shared across all galaxy types
                          (induces cross-correlations between types)
        - 'independent' : one z per pixel per type (no cross-correlations)
        - 'zero'        : no lognormal scatter; reduces to pure Poisson
    sigma_type : {'density', 'constant'}, optional
        How the lognormal scatter sigma varies with density.
        Ignored when z_type='zero'.
        - 'density'  : sigma(r) = S * (r^gamma1 + A_sigma * r^gamma2)
        - 'constant' : sigma is a per-type constant

    Returns
    -------
    model : callable
        NumPyro model with signature model(counts, delta).
    """
    if mean_type not in ('neyrinck', 'powerlaw'):
        raise ValueError("mean_type must be 'neyrinck' or 'powerlaw', got '%s'" % mean_type)
    if z_type not in ('shared', 'independent', 'zero'):
        raise ValueError("z_type must be 'shared', 'independent', or 'zero', got '%s'" % z_type)
    if z_type != 'zero' and sigma_type not in ('density', 'constant'):
        raise ValueError("sigma_type must be 'density' or 'constant', got '%s'" % sigma_type)

    def model(counts, delta):
        N_types, N_pix = counts.shape
        r = 1.0 + delta

        # ---- Mean ----
        if mean_type == 'neyrinck':
            n_bar, beta, delta_g = _sample_neyrinck_params(N_types)
            mu_det = _neyrinck_mean(r, n_bar, beta, delta_g)
        else:
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

        # ---- Shared z: one draw per pixel, same across all types ----
        if z_type == 'shared':
            with numpyro.plate("pix", N_pix):
                z     = numpyro.sample("z", dist.Normal(0.0, 1.0))
                sigma = (_density_sigma(r, S, gamma1, gamma2, A_sigma)
                         if sigma_type == 'density'
                         else jnp.clip(sigma_t[:, None] * jnp.ones((1, N_pix)), 1e-6, 4.0))
                lam   = jnp.exp(sigma * z[None, :] - 0.5 * sigma ** 2)
                _observe(mu_det * lam, counts)

        # ---- Independent z: one draw per type per pixel ----
        else:
            z = numpyro.sample("z", dist.Normal(jnp.zeros((N_types, N_pix)), 1.0))
            with numpyro.plate("pix", N_pix):
                sigma = (_density_sigma(r, S, gamma1, gamma2, A_sigma)
                         if sigma_type == 'density'
                         else jnp.clip(sigma_t[:, None] * jnp.ones((1, N_pix)), 1e-6, 4.0))
                lam   = jnp.exp(sigma * z - 0.5 * sigma ** 2)
                _observe(mu_det * lam, counts)

    return model
