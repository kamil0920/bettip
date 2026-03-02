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


def overdispersed_cdf(k, lam, stat_name: str):
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

    Returns:
        CDF value P(X <= k). Same shape as broadcast(k, lam).
    """
    d = DISPERSION_RATIOS.get(stat_name, 1.0)
    if d <= 1.0:
        return poisson.cdf(k, lam)

    lam = np.asarray(lam, dtype=float)
    k = np.asarray(k, dtype=float)

    p = 1.0 / d
    n = np.where(lam > 0, lam / (d - 1.0), 1.0)

    return nbinom.cdf(k, n, p)
