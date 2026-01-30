"""
Bet-Type-Specific Feature Parameter Configuration Manager

This module provides configuration management for feature engineering parameters
that can be optimized independently per bet type.

Different markets benefit from different temporal parameters:
- away_win predictions may need aggressive ELO (high k_factor)
- fouls predictions may benefit from longer form windows
- BTTS predictions may need different poisson lookback periods
"""
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json

import yaml


# Default feature params directory
FEATURE_PARAMS_DIR = Path("config/feature_params")

def clean_numpy_types(data):
    """Recursively convert NumPy types to native Python types for clean YAML saving."""
    if isinstance(data, dict):
        return {k: clean_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_numpy_types(v) for v in data]
    elif isinstance(data, np.ndarray):
        return clean_numpy_types(data.tolist())
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.generic):
        # Catch-all for any other numpy scalar types (e.g., multiarray.scalar)
        return data.item()
    return data

@dataclass
class BetTypeFeatureConfig:
    """
    Configuration for feature engineering parameters specific to a bet type.

    This dataclass contains all tunable parameters that affect feature generation.
    Each bet type can have its own optimized values discovered via walk-forward validation.

    Attributes:
        bet_type: The bet type this config applies to (e.g., 'away_win', 'fouls')

        Phase 1 - Core parameters (5 high-impact params):
        - elo_k_factor: ELO rating volatility (higher = more reactive)
        - elo_home_advantage: Home team ELO bonus
        - form_window: Number of matches for form calculation
        - ema_span: Exponential moving average span for recency weighting
        - poisson_lookback: Matches to consider for goal rate estimation

        Phase 2 - Extended parameters:
        - half_life_days: Decay rate for Dixon-Coles time weighting

        Niche market parameters (independent per market):
        - fouls_ema_span: EMA span specifically for fouls features
        - cards_ema_span: EMA span specifically for cards features
        - shots_ema_span: EMA span specifically for shots features
        - corners_ema_span: EMA span specifically for corners features

        Metadata:
        - optimized: Whether these params were discovered via optimization
        - optimization_date: When optimization was performed
        - precision: Precision achieved with these params during optimization
        - n_trials: Number of Optuna trials used
    """
    bet_type: str

    # Phase 1: Core parameters (5 key params)
    elo_k_factor: float = 32.0
    elo_home_advantage: float = 100.0
    form_window: int = 5
    ema_span: int = 5
    poisson_lookback: int = 10

    # Phase 2: Extended parameters
    half_life_days: float = 60.0
    h2h_matches: int = 5
    goal_diff_lookback: int = 5
    home_away_form_window: int = 5

    # Niche market-specific EMA spans (independent per market)
    fouls_ema_span: int = 10
    cards_ema_span: int = 10
    shots_ema_span: int = 10
    corners_ema_span: int = 10

    # Niche market window sizes
    fouls_window_sizes: List[int] = field(default_factory=lambda: [5, 10])
    cards_window_sizes: List[int] = field(default_factory=lambda: [5, 10])
    shots_window_sizes: List[int] = field(default_factory=lambda: [5, 10])
    corners_window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Metadata
    optimized: bool = False
    optimization_date: Optional[str] = None
    precision: Optional[float] = None
    roi: Optional[float] = None
    n_trials: Optional[int] = None

    def to_registry_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert to registry parameter format for feature engineer instantiation.

        Returns a dict mapping engineer names to their parameter overrides.
        These params will be merged with default params when creating engineers.

        Returns:
            Dict mapping engineer name to parameter dict
        """
        return {
            'elo': {
                'k_factor': self.elo_k_factor,
                'home_advantage': self.elo_home_advantage,
            },
            'team_form': {
                'n_matches': self.form_window,
            },
            'ema': {
                'span': self.ema_span,
            },
            'poisson': {
                'lookback_matches': self.poisson_lookback,
            },
            'goal_diff': {
                'lookback_matches': self.goal_diff_lookback,
            },
            'h2h': {
                'n_h2h': self.h2h_matches,
            },
            'home_away_form': {
                'n_matches': self.home_away_form_window,
            },
            # Niche market features
            'fouls': {
                'window_sizes': self.fouls_window_sizes,
                'ema_span': self.fouls_ema_span,
            },
            'cards': {
                'window_sizes': self.cards_window_sizes,
                'ema_span': self.cards_ema_span,
            },
            'shots': {
                'window_sizes': self.shots_window_sizes,
                'ema_span': self.shots_ema_span,
            },
            'corners': {
                'window_sizes': self.corners_window_sizes,
                'ema_span': self.corners_ema_span,
            },
        }

    def params_hash(self) -> str:
        """
        Generate a hash of feature parameters for cache key generation.

        Only includes parameters that affect feature generation (not metadata).

        Returns:
            MD5 hash string of the parameter values
        """
        params_dict = {
            'elo_k_factor': self.elo_k_factor,
            'elo_home_advantage': self.elo_home_advantage,
            'form_window': self.form_window,
            'ema_span': self.ema_span,
            'poisson_lookback': self.poisson_lookback,
            'half_life_days': self.half_life_days,
            'h2h_matches': self.h2h_matches,
            'goal_diff_lookback': self.goal_diff_lookback,
            'home_away_form_window': self.home_away_form_window,
            'fouls_ema_span': self.fouls_ema_span,
            'cards_ema_span': self.cards_ema_span,
            'shots_ema_span': self.shots_ema_span,
            'corners_ema_span': self.corners_ema_span,
            'fouls_window_sizes': tuple(self.fouls_window_sizes),
            'cards_window_sizes': tuple(self.cards_window_sizes),
            'shots_window_sizes': tuple(self.shots_window_sizes),
            'corners_window_sizes': tuple(self.corners_window_sizes),
        }
        params_str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:12]

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save configuration to YAML file.

        Args:
            path: Optional path to save to. Defaults to config/feature_params/{bet_type}.yaml

        Returns:
            Path where config was saved
        """
        if path is None:
            FEATURE_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
            path = FEATURE_PARAMS_DIR / f"{self.bet_type}.yaml"

        # Convert to dict, handling list fields
        data = asdict(self)

        clean_data = clean_numpy_types(data)

        with open(path, 'w') as f:
            yaml.dump(clean_data, f, default_flow_style=False, sort_keys=False)

        return path

    @classmethod
    def load(cls, path: Path) -> 'BetTypeFeatureConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            BetTypeFeatureConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def load_for_bet_type(cls, bet_type: str) -> 'BetTypeFeatureConfig':
        """
        Load configuration for a specific bet type.

        Looks for config in standard location: config/feature_params/{bet_type}.yaml
        Falls back to default.yaml if bet-type-specific config doesn't exist.
        Falls back to default values if no config files exist.

        Args:
            bet_type: The bet type to load config for

        Returns:
            BetTypeFeatureConfig instance
        """
        # Try bet-type-specific config first
        bet_type_path = FEATURE_PARAMS_DIR / f"{bet_type}.yaml"
        if bet_type_path.exists():
            return cls.load(bet_type_path)

        # Fall back to default config
        default_path = FEATURE_PARAMS_DIR / "default.yaml"
        if default_path.exists():
            config = cls.load(default_path)
            config.bet_type = bet_type
            return config

        # Return default values
        return cls(bet_type=bet_type)

    def update_metadata(
        self,
        precision: float,
        roi: float,
        n_trials: int,
    ) -> None:
        """
        Update metadata after optimization.

        Args:
            precision: Achieved precision
            roi: Achieved ROI
            n_trials: Number of Optuna trials used
        """
        self.optimized = True
        self.optimization_date = datetime.now().isoformat()
        self.precision = precision
        self.roi = roi
        self.n_trials = n_trials

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            f"BetTypeFeatureConfig: {self.bet_type}",
            f"  Optimized: {self.optimized}",
            "",
            "  Core Parameters:",
            f"    elo_k_factor: {self.elo_k_factor}",
            f"    elo_home_advantage: {self.elo_home_advantage}",
            f"    form_window: {self.form_window}",
            f"    ema_span: {self.ema_span}",
            f"    poisson_lookback: {self.poisson_lookback}",
        ]

        if self.bet_type in ['fouls', 'cards', 'shots', 'corners']:
            lines.extend([
                "",
                "  Niche Market Parameters:",
                f"    {self.bet_type}_ema_span: {getattr(self, f'{self.bet_type}_ema_span')}",
                f"    {self.bet_type}_window_sizes: {getattr(self, f'{self.bet_type}_window_sizes')}",
            ])

        if self.optimized:
            lines.extend([
                "",
                "  Optimization Results:",
                f"    precision: {self.precision:.1%}" if self.precision else "    precision: N/A",
                f"    roi: {self.roi:+.1f}%" if self.roi else "    roi: N/A",
                f"    n_trials: {self.n_trials}",
                f"    date: {self.optimization_date}",
            ])

        return "\n".join(lines)


# Parameter search spaces for Optuna Bayesian optimization
# Format: (min, max, type) where type is 'int' or 'float'
# This enables true Bayesian optimization with TPE sampler
PARAMETER_SEARCH_SPACES = {
    # Phase 1: Core parameters
    # Bounds expanded based on R32/R33 boundary analysis (8 params at boundary)
    'elo_k_factor': (5, 70, 'int'),            # ELO volatility (was 10-50; fouls@48, under25@10, shots@12)
    'elo_home_advantage': (15, 250, 'int'),    # Home advantage points (was 25-250; home_win@47)
    'form_window': (2, 28, 'int'),             # Recent matches for form (was 3-20; fouls@20, over25@3)
    'ema_span': (2, 20, 'int'),                # EMA smoothing window (was 3-20; btts@4, under25@3)
    'poisson_lookback': (5, 40, 'int'),        # Goal rate estimation window

    # Phase 2: Extended parameters
    'half_life_days': (20.0, 150.0, 'float'),  # Time decay half-life
    'h2h_matches': (3, 12, 'int'),             # Head-to-head history
    'goal_diff_lookback': (3, 12, 'int'),      # Goal difference window
    'home_away_form_window': (3, 12, 'int'),   # Venue-specific form

    # Niche market EMA spans
    'fouls_ema_span': (3, 20, 'int'),
    'cards_ema_span': (3, 20, 'int'),
    'shots_ema_span': (3, 20, 'int'),
    'corners_ema_span': (3, 20, 'int'),
}


# Bet type categories and their primary parameters to optimize
BET_TYPE_PARAM_PRIORITIES = {
    # Match result markets
    'away_win': ['elo_k_factor', 'elo_home_advantage', 'form_window', 'ema_span'],
    'home_win': ['elo_k_factor', 'elo_home_advantage', 'form_window', 'ema_span'],

    # Goals markets
    'btts': ['elo_k_factor', 'form_window', 'ema_span', 'poisson_lookback'],
    'over25': ['elo_k_factor', 'form_window', 'ema_span', 'poisson_lookback'],
    'under25': ['elo_k_factor', 'form_window', 'ema_span', 'poisson_lookback'],

    # Niche markets (each gets its own market-specific EMA)
    'fouls': ['elo_k_factor', 'form_window', 'fouls_ema_span'],
    'cards': ['elo_k_factor', 'form_window', 'cards_ema_span'],
    'shots': ['elo_k_factor', 'form_window', 'shots_ema_span'],
    'corners': ['elo_k_factor', 'form_window', 'corners_ema_span'],
}


def get_search_space_for_bet_type(bet_type: str) -> Dict[str, tuple]:
    """
    Get the parameter search space for a specific bet type.

    Only includes parameters that are relevant for the bet type.

    Args:
        bet_type: The bet type

    Returns:
        Dict mapping parameter name to (min, max, type) tuples for Bayesian optimization
    """
    params = BET_TYPE_PARAM_PRIORITIES.get(bet_type, ['elo_k_factor', 'form_window', 'ema_span'])
    return {p: PARAMETER_SEARCH_SPACES[p] for p in params if p in PARAMETER_SEARCH_SPACES}
