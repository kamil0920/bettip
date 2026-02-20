"""
Prediction Health Reporting for Daily Recommendations Pipeline.

Provides structured health tracking per prediction run:
- Per-market status (models loaded, features matched, calibration, odds source)
- Feature mismatch severity classification with tiered response
- Two-stage model fallback when odds unavailable
- Calibration collapse detection with conservative threshold multiplier
- Model mismatch auto-resolution (skip market, don't crash)
- Summary JSON output for debugging

Usage:
    from src.ml.prediction_health import HealthTracker

    tracker = HealthTracker()
    report = tracker.create_market_report("cards_under_35")
    report.record_models_loaded(["cards_under_35_lightgbm", "cards_under_35_xgboost"])
    report.record_feature_match(expected=30, missing=1, missing_names=["feat_x"])
    report.record_calibration(CalibrationStatus.CALIBRATED)
    report.record_odds("real", odds_value=1.85)
    tracker.finalize(report)

    tracker.write_summary(Path("data/05-recommendations/health_20260219.json"))
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketStatus(str, Enum):
    """Overall status for a market prediction."""

    OK = "ok"
    DEGRADED = "degraded"
    SKIPPED = "skipped"


class CalibrationStatus(str, Enum):
    """Calibration state of the model used for prediction."""

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    UNKNOWN = "unknown"


class FeatureMismatchSeverity(str, Enum):
    """Severity tier for feature mismatches.

    Thresholds:
    - NONE: 0% missing
    - LOW: <2% missing — proceed with warning
    - MEDIUM: 2-5% missing — degrade confidence
    - HIGH: >5% missing — skip market
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Conservative threshold multiplier applied when calibration is degenerate.
# Raises the effective threshold by 20% to reduce false positives from
# uncalibrated probabilities.
UNCALIBRATED_THRESHOLD_MULTIPLIER = 1.2

# Confidence penalty applied when feature mismatch is MEDIUM (2-5%).
# Reduces reported confidence to signal degraded prediction quality.
MEDIUM_MISMATCH_CONFIDENCE_PENALTY = 0.85


def classify_feature_mismatch(
    n_expected: int,
    n_missing: int,
) -> FeatureMismatchSeverity:
    """Classify feature mismatch severity based on percentage missing.

    Args:
        n_expected: Total features the model expects.
        n_missing: Number of features not found in inference data.

    Returns:
        Severity tier for the mismatch.
    """
    if n_expected <= 0 or n_missing <= 0:
        return FeatureMismatchSeverity.NONE
    pct = n_missing / n_expected
    if pct < 0.02:
        return FeatureMismatchSeverity.LOW
    elif pct <= 0.05:
        return FeatureMismatchSeverity.MEDIUM
    else:
        return FeatureMismatchSeverity.HIGH


@dataclass
class MarketHealthReport:
    """Health report for a single market prediction.

    Accumulates diagnostic information during the prediction flow
    and resolves a final MarketStatus when finalized.

    ``threshold_multiplier`` (from calibration collapse) and
    ``confidence_penalty`` (from feature mismatch) are intentionally
    independent and do not compound. ``threshold_multiplier`` raises the
    probability bar required to emit a bet, while ``confidence_penalty``
    scales the reported confidence score downward. They target different
    stages of the decision pipeline.
    """

    market: str
    status: MarketStatus = MarketStatus.OK
    models_loaded: List[str] = field(default_factory=list)
    models_missing: List[str] = field(default_factory=list)
    n_features_expected: int = 0
    n_features_missing: int = 0
    missing_feature_names: List[str] = field(default_factory=list)
    feature_mismatch_severity: FeatureMismatchSeverity = FeatureMismatchSeverity.NONE
    feature_mismatch_pct: float = 0.0
    calibration_status: CalibrationStatus = CalibrationStatus.UNKNOWN
    odds_source: str = "none"
    odds_value: Optional[float] = None
    two_stage_fallback: bool = False
    threshold_multiplier: float = 1.0
    confidence_penalty: float = 1.0
    skip_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    conformal_uncertainty: Optional[float] = None
    uncertainty_penalty: float = 1.0

    def record_models_loaded(
        self,
        loaded: List[str],
        missing: Optional[List[str]] = None,
    ) -> None:
        """Record which models were loaded and which were missing.

        Args:
            loaded: Model names that loaded successfully.
            missing: Model names that failed to load (optional).
        """
        self.models_loaded = list(loaded)
        if missing:
            self.models_missing = list(missing)
            for m in missing:
                self.warnings.append(f"Model file missing: {m}")

    def record_feature_match(
        self,
        expected: int,
        missing: int,
        missing_names: Optional[List[str]] = None,
    ) -> None:
        """Record feature match result and classify severity.

        Args:
            expected: Number of features the model expects.
            missing: Number of missing features.
            missing_names: Names of the missing features (truncated to 10).
        """
        self.n_features_expected = expected
        self.n_features_missing = missing
        self.feature_mismatch_pct = missing / expected if expected > 0 else 0.0
        if missing_names:
            self.missing_feature_names = list(missing_names[:10])

        self.feature_mismatch_severity = classify_feature_mismatch(expected, missing)

        if self.feature_mismatch_severity == FeatureMismatchSeverity.LOW:
            self.warnings.append(
                f"Feature mismatch LOW ({missing}/{expected}, "
                f"{self.feature_mismatch_pct:.1%}): proceed with warning"
            )
        elif self.feature_mismatch_severity == FeatureMismatchSeverity.MEDIUM:
            self.confidence_penalty = MEDIUM_MISMATCH_CONFIDENCE_PENALTY
            self.status = MarketStatus.DEGRADED
            self.warnings.append(
                f"Feature mismatch MEDIUM ({missing}/{expected}, "
                f"{self.feature_mismatch_pct:.1%}): confidence degraded"
            )
        elif self.feature_mismatch_severity == FeatureMismatchSeverity.HIGH:
            self.status = MarketStatus.SKIPPED
            self.skip_reason = (
                f"Feature mismatch too high: {missing}/{expected} "
                f"({self.feature_mismatch_pct:.1%} > 5%)"
            )

    def record_calibration(self, status: CalibrationStatus) -> None:
        """Record calibration status and apply threshold multiplier if needed.

        Args:
            status: Detected calibration status.
        """
        self.calibration_status = status
        if status == CalibrationStatus.UNCALIBRATED:
            self.threshold_multiplier = UNCALIBRATED_THRESHOLD_MULTIPLIER
            if self.status != MarketStatus.SKIPPED:
                self.status = MarketStatus.DEGRADED
            self.warnings.append(
                "Calibration degenerate: applying "
                f"{UNCALIBRATED_THRESHOLD_MULTIPLIER}x threshold multiplier"
            )

    def record_odds(
        self,
        source: str,
        odds_value: Optional[float] = None,
    ) -> None:
        """Record odds source and value.

        Args:
            source: One of "real", "baseline", "none".
            odds_value: Decimal odds value (if available).
        """
        self.odds_source = source
        self.odds_value = odds_value

    def record_two_stage_fallback(self) -> None:
        """Record that a two-stage model fell back to Stage 1 only."""
        self.two_stage_fallback = True
        if self.status != MarketStatus.SKIPPED:
            self.status = MarketStatus.DEGRADED
        self.warnings.append(
            "Two-stage model: odds unavailable, using Stage 1 "
            "probability-only prediction with reduced confidence"
        )

    def record_skip(self, reason: str) -> None:
        """Record that this market was skipped entirely.

        Args:
            reason: Human-readable reason for skipping.
        """
        self.status = MarketStatus.SKIPPED
        self.skip_reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "market": self.market,
            "status": self.status.value,
            "models_loaded": self.models_loaded,
            "models_missing": self.models_missing,
            "features": {
                "expected": self.n_features_expected,
                "missing": self.n_features_missing,
                "mismatch_pct": round(self.feature_mismatch_pct, 4),
                "severity": self.feature_mismatch_severity.value,
                "missing_names": self.missing_feature_names,
            },
            "calibration_status": self.calibration_status.value,
            "odds_source": self.odds_source,
            "odds_value": self.odds_value,
            "two_stage_fallback": self.two_stage_fallback,
            "threshold_multiplier": self.threshold_multiplier,
            "confidence_penalty": self.confidence_penalty,
            "skip_reason": self.skip_reason,
            "warnings": self.warnings,
            "conformal_uncertainty": self.conformal_uncertainty,
            "uncertainty_penalty": self.uncertainty_penalty,
        }


class HealthTracker:
    """Accumulates per-market health reports for a prediction run.

    Usage:
        tracker = HealthTracker()
        report = tracker.create_market_report("cards_under_35")
        # ... populate report ...
        tracker.finalize(report)
        tracker.write_summary(output_path)
    """

    def __init__(self) -> None:
        self._reports: Dict[str, MarketHealthReport] = {}
        self._run_timestamp: str = datetime.utcnow().isoformat()
        self._global_warnings: List[str] = []

    def create_market_report(self, market: str) -> MarketHealthReport:
        """Create a new health report for a market.

        Args:
            market: Market name (e.g. "cards_under_35").

        Returns:
            A new MarketHealthReport instance.
        """
        return MarketHealthReport(market=market)

    def finalize(self, report: MarketHealthReport) -> None:
        """Finalize and store a market health report.

        If the same market is finalized multiple times (e.g. across
        multiple matches), the latest report is kept.

        Args:
            report: The completed market health report.
        """
        key = report.market
        if key in self._reports:
            # Merge: keep the worse status
            existing = self._reports[key]
            _STATUS_SEVERITY = {
                MarketStatus.OK: 0,
                MarketStatus.DEGRADED: 1,
                MarketStatus.SKIPPED: 2,
            }
            if _STATUS_SEVERITY.get(report.status, 0) > _STATUS_SEVERITY.get(existing.status, 0):
                existing.status = report.status
                existing.skip_reason = existing.skip_reason or report.skip_reason
            existing.warnings.extend(report.warnings)
        else:
            self._reports[key] = report

        if report.warnings:
            for w in report.warnings:
                logger.info(f"[HEALTH] {report.market}: {w}")

    def add_global_warning(self, warning: str) -> None:
        """Add a warning that applies to the entire run.

        Args:
            warning: Warning message.
        """
        self._global_warnings.append(warning)
        logger.warning(f"[HEALTH] {warning}")

    def get_report(self, market: str) -> Optional[MarketHealthReport]:
        """Get the health report for a specific market.

        Args:
            market: Market name.

        Returns:
            The report, or None if not tracked.
        """
        return self._reports.get(market)

    def summary(self) -> Dict[str, Any]:
        """Generate summary dict of all market health reports.

        Returns:
            Dict with run metadata and per-market status.
        """
        markets_ok = sum(1 for r in self._reports.values() if r.status == MarketStatus.OK)
        markets_degraded = sum(
            1 for r in self._reports.values() if r.status == MarketStatus.DEGRADED
        )
        markets_skipped = sum(1 for r in self._reports.values() if r.status == MarketStatus.SKIPPED)

        return {
            "schema_version": "1.0",
            "run_timestamp": self._run_timestamp,
            "total_markets": len(self._reports),
            "markets_ok": markets_ok,
            "markets_degraded": markets_degraded,
            "markets_skipped": markets_skipped,
            "global_warnings": self._global_warnings,
            "markets": {name: report.to_dict() for name, report in sorted(self._reports.items())},
        }

    def write_summary(self, output_path: Path) -> None:
        """Write summary JSON to disk.

        Args:
            output_path: Path for the output JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        # Atomic write: write to temp file, then rename to avoid
        # corrupted JSON if the process crashes mid-write.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
        try:
            with open(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(output_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        logger.info(
            f"[HEALTH] Summary written to {output_path} "
            f"(ok={data['markets_ok']}, degraded={data['markets_degraded']}, "
            f"skipped={data['markets_skipped']})"
        )
