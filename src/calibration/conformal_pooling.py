"""
Scale-normalized cross-market conformal pooling.

Data-poor markets borrow statistical strength from data-rich markets
by normalizing conformal residuals to a common scale, pooling, and
computing a single global quantile — then rescaling per-market.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES_PER_MARKET: int = 10


class CrossMarketConformalPooler:
    """Pool conformal residuals across betting markets for robust quantile estimation.

    Markets with few calibration samples produce noisy conformal thresholds.
    This class normalises residuals by per-market MAE so they share a common
    scale, pools them into a single distribution, and extracts a global
    quantile that is then rescaled back to each market's native scale.

    Workflow::

        pooler = CrossMarketConformalPooler()
        pooler.add_market("over25", preds_o25, actuals_o25)
        pooler.add_market("btts", preds_btts, actuals_btts)
        pooler.fit(alpha=0.10)

        tau_over25 = pooler.get_tau("over25")
        tau_btts   = pooler.get_tau("btts")

    Attributes:
        markets: Mapping of market name to raw predictions/actuals.
        scales: Per-market MAE computed during :meth:`fit`.
        global_tau_z: Global normalised quantile (set after :meth:`fit`).
        alpha: Significance level used during :meth:`fit`.
    """

    def __init__(self) -> None:
        self.markets: Dict[str, Dict[str, np.ndarray]] = {}
        self.scales: Dict[str, float] = {}
        self.global_tau_z: Optional[float] = None
        self.alpha: Optional[float] = None

    # ------------------------------------------------------------------
    # Data accumulation
    # ------------------------------------------------------------------

    def add_market(
        self,
        name: str,
        preds: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        """Register a market's calibration predictions and actuals.

        Args:
            name: Unique market identifier (e.g. ``"over25"``, ``"btts"``).
            preds: 1-D array of predicted probabilities.
            actuals: 1-D array of binary outcomes (0/1).

        Raises:
            ValueError: If arrays have mismatched lengths or fewer than
                :data:`MIN_SAMPLES_PER_MARKET` observations.
        """
        preds = np.asarray(preds, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()

        if preds.shape[0] != actuals.shape[0]:
            raise ValueError(
                f"Market '{name}': preds length {preds.shape[0]} != "
                f"actuals length {actuals.shape[0]}"
            )
        if preds.shape[0] < MIN_SAMPLES_PER_MARKET:
            raise ValueError(
                f"Market '{name}': only {preds.shape[0]} samples, "
                f"need >= {MIN_SAMPLES_PER_MARKET}"
            )

        self.markets[name] = {"preds": preds, "actuals": actuals}
        logger.info("Added market '%s' with %d samples", name, preds.shape[0])

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, alpha: float = 0.10) -> CrossMarketConformalPooler:
        """Normalise residuals by per-market MAE, pool, and compute global quantile.

        Steps:

        1. Per market *i*: ``residuals_i = preds_i - actuals_i``,
           ``scale_i = MAE(preds_i, actuals_i)``.
        2. Normalise: ``z_i = residuals_i / scale_i``.
        3. Pool all ``z_i`` across markets.
        4. Global ``tau_z = z_{(k)}`` where
           ``k = ceil((n + 1) * (1 - alpha))`` (finite-sample correction).
        5. Per-market ``tau = tau_z * scale_i`` (retrieved via :meth:`get_tau`).

        Args:
            alpha: Significance level. ``1 - alpha`` is the coverage target
                (default ``0.10`` → 90 % coverage).

        Returns:
            self (for chaining).

        Raises:
            RuntimeError: If no markets have been added.
            ValueError: If any per-market MAE is effectively zero.
        """
        if not self.markets:
            raise RuntimeError("No markets added — call add_market() first.")

        self.alpha = alpha
        pooled_z: List[np.ndarray] = []

        for name, data in self.markets.items():
            residuals = data["preds"] - data["actuals"]
            scale = float(np.mean(np.abs(residuals)))

            if scale < 1e-12:
                raise ValueError(
                    f"Market '{name}' has near-zero MAE ({scale:.2e}). "
                    "Predictions may be trivially perfect — cannot normalise."
                )

            self.scales[name] = scale
            z = residuals / scale
            pooled_z.append(z)
            logger.info(
                "Market '%s': n=%d, MAE=%.4f",
                name,
                len(residuals),
                scale,
            )

        all_z = np.concatenate(pooled_z)
        n = len(all_z)

        # Finite-sample conformal quantile (order statistic)
        k = math.ceil((n + 1) * (1 - alpha))
        k = min(k, n)  # clamp to array length

        # One-sided quantile: positive z = overconfident prediction
        sorted_z = np.sort(all_z)
        self.global_tau_z = float(sorted_z[k - 1])  # 1-indexed → 0-indexed

        logger.info(
            "Pooled %d residuals across %d markets. "
            "Global tau_z=%.4f at alpha=%.2f (k=%d)",
            n,
            len(self.markets),
            self.global_tau_z,
            alpha,
            k,
        )

        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_tau(self, market_name: str) -> float:
        """Return the rescaled conformal threshold for a specific market.

        ``tau = global_tau_z * scale_i``

        Args:
            market_name: Market identifier previously passed to :meth:`add_market`.

        Returns:
            Per-market conformal half-width in the original probability scale.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            KeyError: If *market_name* was not added.
        """
        if self.global_tau_z is None:
            raise RuntimeError("Call fit() before get_tau().")
        if market_name not in self.scales:
            raise KeyError(
                f"Market '{market_name}' not found. "
                f"Available: {list(self.scales.keys())}"
            )

        return self.global_tau_z * self.scales[market_name]

    def get_tau_per_market(self) -> Dict[str, float]:
        """Return the rescaled conformal threshold for all fitted markets.

        Returns:
            Dictionary mapping market name to its conformal tau.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.global_tau_z is None:
            raise RuntimeError("Call fit() before get_tau_per_market().")
        return {name: self.get_tau(name) for name in self.scales}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise fitted state to a plain dictionary.

        Only stores the fitted artefacts (scales, global quantile, alpha),
        **not** the raw predictions/actuals.

        Returns:
            Dictionary suitable for JSON / joblib persistence.
        """
        return {
            "scales": dict(self.scales),
            "global_tau_z": self.global_tau_z,
            "alpha": self.alpha,
            "n_markets": len(self.scales),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CrossMarketConformalPooler:
        """Reconstruct a fitted pooler from a serialised dictionary.

        Args:
            d: Dictionary produced by :meth:`to_dict`.

        Returns:
            A :class:`CrossMarketConformalPooler` ready for :meth:`get_tau` calls.

        Raises:
            KeyError: If required keys are missing.
        """
        pooler = cls()
        pooler.scales = dict(d["scales"])
        pooler.global_tau_z = d["global_tau_z"]
        pooler.alpha = d["alpha"]
        return pooler
