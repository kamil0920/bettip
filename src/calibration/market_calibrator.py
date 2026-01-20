"""
Market-specific probability calibration.

This module provides probability calibration for betting callibration to address
overconfidence issues identified in paper trading validation.

Calibration methods:
1. Temperature scaling - applies T parameter to logits
2. Linear calibration - applies a simple multiplier
3. Platt scaling - learns a,b parameters for sigmoid(a*p + b)

Paper Trading Results (Jan 17-19, 2026):
- FOULS: 81.8% actual vs 69.8% predicted -> T=0.45 (underconfident)
- BTTS: 42.9% actual vs 64% predicted -> T=10.0 (overconfident)
- HOME_WIN: 50% actual vs 76% predicted -> T=10.0 (overconfident)
- OVER_2.5: 30% actual vs 70% predicted -> T=10.0 (severely overconfident)
- SHOTS: 50% actual vs 75% predicted -> T=10.0 (overconfident)
- CORNERS: 0% actual vs 63% predicted -> T=10.0 (severely overconfident)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import numpy as np
import yaml
from pathlib import Path


@dataclass
class CalibrationConfig:
    """Configuration for a single market's calibration."""
    factor: float = 1.0  # Linear calibration factor
    temperature: float = 1.0  # Temperature scaling parameter
    enabled: bool = True
    min_threshold: float = 0.5  # Minimum probability threshold after calibration
    method: str = "linear"  # "linear", "temperature", or "platt"
    platt_a: float = 1.0  # Platt scaling parameters
    platt_b: float = 0.0


# Default calibration factors based on paper trading validation
# These should be updated as more data is collected
DEFAULT_CALIBRATION: Dict[str, CalibrationConfig] = {
    'FOULS': CalibrationConfig(
        factor=1.0,  # Model is underconfident - no reduction needed
        temperature=1.0,
        enabled=True,
        min_threshold=0.70,
        method="linear"
    ),
    'SHOTS': CalibrationConfig(
        factor=0.92,  # Backtest: -5.6% gap (71% pred → 65.3% actual)
        temperature=1.2,
        enabled=True,
        min_threshold=0.70,
        method="linear"
    ),
    'CORNERS': CalibrationConfig(
        factor=0.88,  # Backtest: -8.2% gap (71% pred → 63% actual)
        temperature=1.5,
        enabled=True,
        min_threshold=0.70,
        method="linear"
    ),
    'HOME_WIN': CalibrationConfig(
        factor=0.7,  # Severe overconfidence
        temperature=3.0,
        enabled=False,  # Disabled until validated
        min_threshold=0.70,
        method="linear"
    ),
    'AWAY_WIN': CalibrationConfig(
        factor=0.97,  # Backtest: -2.2% gap (69% pred → 66.8% actual)
        temperature=1.1,
        enabled=True,
        min_threshold=0.55,
        method="linear"
    ),
    'BTTS': CalibrationConfig(
        factor=0.90,  # Backtest: -6.5% gap at 0.65 threshold (67% pred → 60.5% actual)
        temperature=1.5,
        enabled=True,
        min_threshold=0.70,
        method="linear"
    ),
    'OVER_2.5': CalibrationConfig(
        factor=0.5,  # Extremely overconfident - halve probabilities
        temperature=5.0,
        enabled=False,  # DISABLED - 30% hit rate is unacceptable
        min_threshold=0.85,
        method="linear"
    ),
    'UNDER_2.5': CalibrationConfig(
        factor=1.0,  # Unknown - needs validation
        temperature=1.0,
        enabled=True,
        min_threshold=0.70,
        method="linear"
    ),
}


class MarketCalibrator:
    """
    Calibrates model probabilities for different betting markets.

    Usage:
        calibrator = MarketCalibrator()
        calibrator.load_config('config/strategies.yaml')

        # Calibrate a single probability
        raw_prob = 0.75
        calibrated = calibrator.calibrate('BTTS', raw_prob)

        # Calibrate array of probabilities
        probs = np.array([0.65, 0.72, 0.58])
        calibrated = calibrator.calibrate('FOULS', probs)
    """

    def __init__(self, config: Optional[Dict[str, CalibrationConfig]] = None):
        """
        Initialize calibrator with market configurations.

        Args:
            config: Dict mapping market names to CalibrationConfig.
                    Uses DEFAULT_CALIBRATION if not provided.
        """
        self.config = config or DEFAULT_CALIBRATION.copy()

    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load calibration configuration from YAML file.

        Args:
            config_path: Path to config file (typically config/strategies.yaml)
        """
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Load from calibration section
        cal_config = yaml_config.get('calibration', {})
        if not cal_config.get('enabled', True):
            print("Calibration disabled in config")
            return

        per_market = cal_config.get('per_market', {})
        for market, settings in per_market.items():
            market_upper = market.upper()
            if market_upper not in self.config:
                self.config[market_upper] = CalibrationConfig()

            cfg = self.config[market_upper]
            cfg.factor = settings.get('factor', cfg.factor)
            cfg.enabled = settings.get('enabled', cfg.enabled)
            cfg.min_threshold = settings.get('min_threshold', cfg.min_threshold)

        # Also check strategy-specific settings
        strategies = yaml_config.get('strategies', {})
        for strategy_name, strategy_config in strategies.items():
            market = strategy_name.upper()
            if 'calibration_factor' in strategy_config:
                if market not in self.config:
                    self.config[market] = CalibrationConfig()
                self.config[market].factor = strategy_config['calibration_factor']

    def calibrate(
        self,
        market: str,
        probability: Union[float, np.ndarray],
        raw: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Calibrate probability for a specific market.

        Args:
            market: Market name (e.g., 'BTTS', 'FOULS', 'OVER_2.5')
            probability: Raw probability or array of probabilities (0-1)
            raw: If True, skip calibration and return input unchanged

        Returns:
            Calibrated probability capped at [0, 1]
        """
        if raw:
            return probability

        market_upper = market.upper()

        # Handle variations in market naming
        market_map = {
            'OVER25': 'OVER_2.5',
            'UNDER25': 'UNDER_2.5',
            'HOMEWIN': 'HOME_WIN',
            'AWAYWIN': 'AWAY_WIN',
        }
        market_upper = market_map.get(market_upper, market_upper)

        config = self.config.get(market_upper)
        if config is None:
            # Unknown market - return unchanged
            return probability

        if not config.enabled:
            # Market disabled - still calibrate but warn
            pass

        # Apply calibration based on method
        if config.method == "temperature":
            calibrated = self._temperature_scale(probability, config.temperature)
        elif config.method == "platt":
            calibrated = self._platt_scale(probability, config.platt_a, config.platt_b)
        else:  # linear
            calibrated = probability * config.factor

        # Cap at [0, 1]
        if isinstance(calibrated, np.ndarray):
            calibrated = np.clip(calibrated, 0, 1)
        else:
            calibrated = max(0, min(1, calibrated))

        return calibrated

    def _temperature_scale(
        self,
        probability: Union[float, np.ndarray],
        temperature: float
    ) -> Union[float, np.ndarray]:
        """
        Apply temperature scaling to probability.

        Temperature > 1: softens predictions (reduces overconfidence)
        Temperature < 1: sharpens predictions (increases confidence)
        Temperature = 1: no change

        Formula: p_calibrated = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        """
        if temperature == 1.0:
            return probability

        # Avoid division by zero
        eps = 1e-10
        p = np.clip(probability, eps, 1 - eps)

        # Convert to logits, scale, convert back
        p_scaled = np.power(p, 1 / temperature)
        q_scaled = np.power(1 - p, 1 / temperature)

        return p_scaled / (p_scaled + q_scaled)

    def _platt_scale(
        self,
        probability: Union[float, np.ndarray],
        a: float,
        b: float
    ) -> Union[float, np.ndarray]:
        """
        Apply Platt scaling: sigmoid(a * logit(p) + b)
        """
        eps = 1e-10
        p = np.clip(probability, eps, 1 - eps)

        # Convert to logit
        logit = np.log(p / (1 - p))

        # Scale
        scaled_logit = a * logit + b

        # Convert back
        return 1 / (1 + np.exp(-scaled_logit))

    def get_min_threshold(self, market: str) -> float:
        """Get minimum probability threshold for a market."""
        market_upper = market.upper()
        config = self.config.get(market_upper)
        return config.min_threshold if config else 0.5

    def is_enabled(self, market: str) -> bool:
        """Check if a market is enabled for betting."""
        market_upper = market.upper()
        config = self.config.get(market_upper)
        return config.enabled if config else True

    def get_calibration_factor(self, market: str) -> float:
        """Get the calibration factor for a market."""
        market_upper = market.upper()
        config = self.config.get(market_upper)
        return config.factor if config else 1.0


# Convenience function for quick calibration
def calibrate_probability(
    market: str,
    probability: Union[float, np.ndarray],
    config_path: Optional[str] = None
) -> Union[float, np.ndarray]:
    """
    Convenience function to calibrate probability for a market.

    Args:
        market: Market name
        probability: Raw probability
        config_path: Optional path to config file

    Returns:
        Calibrated probability
    """
    calibrator = MarketCalibrator()
    if config_path:
        calibrator.load_config(config_path)
    return calibrator.calibrate(market, probability)
