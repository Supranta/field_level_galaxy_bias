import numpy as np
import jax.numpy as jnp


def neyrinck_model(r, n_bar, beta, delta_g):
    """Neyrinck mean model: n_bar * r^beta * exp(-rho_g / r).

    Parameters
    ----------
    r : array_like
        Local density ratio r = 1 + delta.
    n_bar : array_like
        Mean galaxy count normalisation.
    beta : array_like
        Power-law slope.
    delta_g : array_like
        Void exclusion scale (typically in (-1, 0)).

    Returns
    -------
    mu : array_like
    """
    rho_g = 1.0 + delta_g
    return n_bar * r ** beta * np.exp(-rho_g / r)


def sigma_model(r, sigma1, gamma1, sigma2, gamma2):
    """Two-power-law lognormal scatter model.

    sigma(r) = sigma1 * r^gamma1 + sigma2 * r^gamma2

    Parameters
    ----------
    r : array_like
        Local density ratio r = 1 + delta.
    sigma1, sigma2 : array_like
        Amplitude of each power-law term.
    gamma1, gamma2 : array_like
        Exponents.

    Returns
    -------
    sigma : array_like
    """
    return sigma1 * r ** gamma1 + sigma2 * r ** gamma2


def neyrinck_model_jax(r, n_bar, beta, delta_g):
    """JAX version of the Neyrinck mean model (safe for use inside NumPyro).

    Clamps the exponent to -10 to avoid numerical underflow in voids.
    """
    rho_g          = 1.0 + delta_g
    void_exp_term  = jnp.maximum(-rho_g / r, -10.0)
    return n_bar * r ** beta * jnp.exp(void_exp_term)


def sigma_model_jax(r, S, gamma1, gamma2, A_sigma):
    """JAX version of the lognormal scatter model, reparametrised as:

        sigma(r) = S * (r^gamma1 + A_sigma * r^gamma2)

    where S is a per-type amplitude scale and A_sigma = sigma2 / sigma1
    is a shared ratio between the two power-law terms.
    """
    return S * (r ** gamma1 + A_sigma * r ** gamma2)
