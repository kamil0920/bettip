"""Pre-deployment gate enforcement for sniper_deployment.json.

Validates all enabled markets against deployment criteria from agentspec.json:
- ECE < 0.10 (calibration quality)
- n_bets >= 20 (statistical significance)
- Model files exist (deployment readiness)
- source_run provenance (traceability)

Usage:
    python -m src.ml.deployment_validator [--config PATH] [--auto-fix] [--output-json PATH]
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Gate thresholds from agentspec.json project_invariants
MAX_ECE = 0.10
MIN_HOLDOUT_BETS = 20
MAX_SEED_GAP_PP = 30.0


@dataclass
class MarketValidationResult:
    """Validation result for a single market."""

    market: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregate validation report for all markets."""

    results: list[MarketValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def n_warnings(self) -> int:
        return sum(len(r.warnings) for r in self.results)


def _get_holdout_ece(market_config: dict) -> Optional[float]:
    """Extract ECE from holdout_metrics, falling back to top-level ece."""
    hm = market_config.get("holdout_metrics")
    if isinstance(hm, dict):
        ece = hm.get("ece")
        if ece is not None:
            return float(ece)
    # Fallback to top-level ece
    ece = market_config.get("ece")
    if ece is not None:
        return float(ece)
    return None


def _get_holdout_n_bets(market_config: dict) -> Optional[int]:
    """Extract n_bets from holdout_metrics."""
    hm = market_config.get("holdout_metrics")
    if isinstance(hm, dict):
        n_bets = hm.get("n_bets")
        if n_bets is not None:
            return int(n_bets)
    return None


def _validate_market(
    market_name: str,
    market_config: dict,
    max_ece: float = MAX_ECE,
    min_n_bets: int = MIN_HOLDOUT_BETS,
    models_dir: Optional[Path] = None,
) -> MarketValidationResult:
    """Validate a single enabled market against deployment gates."""
    result = MarketValidationResult(market=market_name, passed=True)

    hm = market_config.get("holdout_metrics")
    if not isinstance(hm, dict):
        result.violations.append("no holdout_metrics")
        result.passed = False
        return result

    # Gate 1: ECE < max_ece
    ece = _get_holdout_ece(market_config)
    if ece is not None and ece > max_ece:
        result.violations.append(f"ECE {ece:.3f} > {max_ece}")
        result.passed = False

    # Gate 2: n_bets >= min_n_bets
    n_bets = _get_holdout_n_bets(market_config)
    if n_bets is not None and n_bets < min_n_bets:
        result.violations.append(f"n_bets {n_bets} < {min_n_bets}")
        result.passed = False

    # Warning: model files exist
    if models_dir:
        for model_path in market_config.get("saved_models", []):
            filename = Path(model_path).name
            if not (models_dir / filename).exists():
                result.warnings.append(f"model file missing: {filename}")

    # Warning: source_run provenance
    source_run = market_config.get("source_run", "")
    if not source_run or source_run == "?":
        result.warnings.append("source_run missing or unknown")

    return result


def validate_deployment_config(
    config_path: Path,
    max_ece: float = MAX_ECE,
    min_n_bets: int = MIN_HOLDOUT_BETS,
    models_dir: Optional[Path] = None,
) -> ValidationReport:
    """Validate all enabled markets in a deployment config.

    Args:
        config_path: Path to sniper_deployment.json.
        max_ece: Maximum allowed ECE (default 0.10).
        min_n_bets: Minimum required holdout bets (default 20).
        models_dir: Optional path to models directory for file checks.

    Returns:
        ValidationReport with per-market results.
    """
    with open(config_path) as f:
        content = f.read()

    # Handle NaN/Infinity in JSON (sniper_deployment.json may contain these)
    content = re.sub(r"\bNaN\b", "null", content)
    content = re.sub(r"\bInfinity\b", "null", content)
    content = re.sub(r"\b-Infinity\b", "null", content)
    config = json.loads(content)

    report = ValidationReport()
    markets = config.get("markets", {})

    for market_name, market_config in markets.items():
        if not market_config.get("enabled", False):
            continue

        result = _validate_market(
            market_name, market_config, max_ece, min_n_bets, models_dir
        )
        report.results.append(result)

    return report


def auto_fix(
    config_path: Path,
    max_ece: float = MAX_ECE,
    min_n_bets: int = MIN_HOLDOUT_BETS,
) -> tuple[ValidationReport, int]:
    """Validate and auto-disable failing markets.

    Returns:
        Tuple of (report, n_disabled).
    """
    with open(config_path) as f:
        content = f.read()

    content = re.sub(r"\bNaN\b", "null", content)
    content = re.sub(r"\bInfinity\b", "null", content)
    content = re.sub(r"\b-Infinity\b", "null", content)
    config = json.loads(content)

    report = ValidationReport()
    markets = config.get("markets", {})
    n_disabled = 0

    for market_name, market_config in markets.items():
        if not market_config.get("enabled", False):
            continue

        result = _validate_market(market_name, market_config, max_ece, min_n_bets)
        report.results.append(result)

        if not result.passed:
            market_config["enabled"] = False
            market_config["disabled_reason"] = (
                f"auto-fix: {'; '.join(result.violations)}"
            )
            n_disabled += 1

    if n_disabled > 0:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    return report, n_disabled


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate deployment config against deployment gates"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/sniper_deployment.json"),
        help="Path to sniper_deployment.json",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Auto-disable markets failing gates",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write validation report as JSON",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Path to models directory for file checks",
    )
    parser.add_argument(
        "--max-ece",
        type=float,
        default=MAX_ECE,
        help=f"Maximum ECE threshold (default {MAX_ECE})",
    )
    parser.add_argument(
        "--min-n-bets",
        type=int,
        default=MIN_HOLDOUT_BETS,
        help=f"Minimum holdout bets (default {MIN_HOLDOUT_BETS})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.config.exists():
        logger.error(f"Config not found: {args.config}")
        return 1

    models_dir = args.models_dir if args.models_dir.exists() else None

    if args.auto_fix:
        report, n_disabled = auto_fix(
            args.config, args.max_ece, args.min_n_bets
        )
        if n_disabled > 0:
            logger.info(f"\nAuto-fix: disabled {n_disabled} market(s)")
    else:
        report = validate_deployment_config(
            args.config, args.max_ece, args.min_n_bets, models_dir
        )

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("DEPLOYMENT VALIDATION REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Gates: ECE < {args.max_ece}, n_bets >= {args.min_n_bets}")
    logger.info(f"{'='*60}\n")

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        logger.info(f"[{status}] {r.market}")
        for v in r.violations:
            logger.info(f"  VIOLATION: {v}")
        for w in r.warnings:
            logger.info(f"  WARNING: {w}")

    logger.info(f"\n{'='*60}")
    logger.info(
        f"Summary: {report.n_passed} passed, {report.n_failed} failed, "
        f"{report.n_warnings} warnings"
    )
    logger.info(f"{'='*60}")

    # Write JSON report
    if args.output_json:
        report_data = {
            "config_path": str(args.config),
            "max_ece": args.max_ece,
            "min_n_bets": args.min_n_bets,
            "n_passed": report.n_passed,
            "n_failed": report.n_failed,
            "n_warnings": report.n_warnings,
            "results": [
                {
                    "market": r.market,
                    "passed": r.passed,
                    "violations": r.violations,
                    "warnings": r.warnings,
                }
                for r in report.results
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"\nReport written to {args.output_json}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
