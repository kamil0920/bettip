"""Overdispersed count distribution CDF for per-line odds estimation.

Sports count statistics (cards, corners, shots, fouls) exhibit overdispersion
(variance > mean), making Poisson CDF a poor fit. This module provides a
Negative Binomial CDF that correctly handles the heavier tails, falling back
to Poisson when dispersion ratio <= 1.0 or the stat is unknown.

Dispersion ratios computed from historical data (all 10 leagues, 2020-2026).
"""

import numpy as np
from scipy.stats import nbinom, poisson

# Empirical dispersion ratios (variance / mean) from historical match data
DISPERSION_RATIOS = {
    "cards": 2.06,
    "corners": 1.35,
    "shots": 1.48,
    "fouls": 1.55,
    "goals": 1.20,
    "ht": 1.10,
}


def match_varying_dispersion(
    stat_name: str,
    goal_supremacy: np.ndarray,
    base_d: float = None,
) -> np.ndarray:
    """Compute per-match dispersion ratio from goal supremacy.

    More one-sided matches (high |goal_supremacy|) exhibit different
    clustering patterns for count statistics (Yip et al.).

    Model: log(d_i) = log(base_d) + alpha_1 * log(1 + |SUP_i|)

    For balanced matches (SUP = 0): d = base_d
    For one-sided matches (|SUP| > 1): d increases (more clustering)

    Args:
        stat_name: Stat identifier for base dispersion lookup.
        goal_supremacy: Array of implied goal supremacy values.
        base_d: Override for base dispersion ratio. If None, uses
                DISPERSION_RATIOS lookup.

    Returns:
        Array of per-match dispersion ratios, same length as goal_supremacy.
    """
    if base_d is None:
        base_d = DISPERSION_RATIOS.get(stat_name, 1.0)

    goal_supremacy = np.asarray(goal_supremacy, dtype=float)

    # Empirical alpha_1 values per stat (calibrated from paper / historical data)
    # Positive alpha_1 means one-sided matches have higher dispersion
    ALPHA_1 = {
        "corners": 0.15,   # Yip et al.: stronger effect for corners
        "cards": 0.10,
        "shots": 0.08,
        "fouls": 0.05,
        "goals": 0.12,
    }
    alpha_1 = ALPHA_1.get(stat_name, 0.0)

    if alpha_1 == 0.0 or base_d <= 1.0:
        return np.full_like(goal_supremacy, base_d)

    # log(d_i) = log(base_d) + alpha_1 * log(1 + |SUP|)
    # At SUP=0: d = base_d. As |SUP| grows: d increases.
    log_d = np.log(base_d) + alpha_1 * np.log1p(np.abs(goal_supremacy))

    # Clamp to reasonable range [1.01, 5.0]
    d = np.exp(log_d)
    d = np.clip(d, 1.01, 5.0)

    return d


def overdispersed_cdf(k, lam, stat_name: str, dispersion: np.ndarray = None):
    """CDF using Negative Binomial when overdispersed, Poisson otherwise.

    NB parametrization (scipy convention):
        p = 1/d, n = lam/(d-1)
    where d = dispersion_ratio (var/mean).

    This gives E[X] = n(1-p)/p = lam and Var[X] = n(1-p)/p^2 = lam*d,
    matching observed overdispersion.

    Falls back to Poisson when d <= 1.0 or stat unknown.

    Args:
        k: Count threshold (scalar or array).
        lam: Expected count / rate parameter (scalar or array).
        stat_name: Stat identifier (e.g., "cards", "corners").
        dispersion: Optional per-match dispersion array. If provided,
                   overrides the fixed DISPERSION_RATIOS lookup.
                   Must broadcast with k and lam.

    Returns:
        CDF value P(X <= k). Same shape as broadcast(k, lam).
    """
    lam = np.asarray(lam, dtype=float)
    k = np.asarray(k, dtype=float)

    if dispersion is not None:
        d = np.asarray(dispersion, dtype=float)
    else:
        d = DISPERSION_RATIOS.get(stat_name, 1.0)

    # Scalar d <= 1.0: use Poisson
    if np.isscalar(d) and d <= 1.0:
        return poisson.cdf(k, lam)

    # Per-match: use NB where d > 1, Poisson where d <= 1
    if not np.isscalar(d):
        result = np.empty_like(lam)
        mask_nb = d > 1.0
        mask_pois = ~mask_nb

        if mask_pois.any():
            result[mask_pois] = poisson.cdf(k[mask_pois] if k.shape else k, lam[mask_pois])
        if mask_nb.any():
            d_nb = d[mask_nb]
            lam_nb = lam[mask_nb]
            k_nb = k[mask_nb] if k.shape else k
            p_nb = 1.0 / d_nb
            n_nb = np.where(lam_nb > 0, lam_nb / (d_nb - 1.0), 1.0)
            result[mask_nb] = nbinom.cdf(k_nb, n_nb, p_nb)
        return result

    p = 1.0 / d
    n = np.where(lam > 0, lam / (d - 1.0), 1.0)

    return nbinom.cdf(k, n, p)
