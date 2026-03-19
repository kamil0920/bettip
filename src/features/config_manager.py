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
import re

import yaml


# Default feature params directory
FEATURE_PARAMS_DIR = Path("config/feature_params")

# Markets where rolling z-score normalization is enabled by default.
# S12 A/B test showed normalization helps niche markets (fouls +29pp, btts +11pp,
# corners +10pp) but severely hurts H2H markets (home_win -92pp, over25 -84pp).
def _yaml_keys(path: Path) -> set:
    """Return the top-level keys present in a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return set(data.keys()) if isinstance(data, dict) else set()


NICHE_NORMALIZE_MARKETS = {
    'fouls', 'cards', 'shots', 'corners', 'btts',
    # Cards (1.5-6.5)
    'cards_over_15', 'cards_over_25', 'cards_over_35',
    'cards_over_45', 'cards_over_55', 'cards_over_65',
    'cards_under_15', 'cards_under_25', 'cards_under_35',
    'cards_under_45', 'cards_under_55', 'cards_under_65',
    # Corners (8.5-11.5)
    'corners_over_85', 'corners_over_95', 'corners_over_105', 'corners_over_115',
    'corners_under_85', 'corners_under_95', 'corners_under_105', 'corners_under_115',
    # Shots (22.5-27.5)
    'shots_over_225', 'shots_over_235', 'shots_over_245', 'shots_over_255', 'shots_over_265', 'shots_over_275',
    'shots_under_225', 'shots_under_235', 'shots_under_245', 'shots_under_255', 'shots_under_265', 'shots_under_275',
    # Fouls (22.5-27.5)
    'fouls_over_225', 'fouls_over_235', 'fouls_over_245', 'fouls_over_255', 'fouls_over_265', 'fouls_over_275',
    'fouls_under_225', 'fouls_under_235', 'fouls_under_245', 'fouls_under_255', 'fouls_under_265', 'fouls_under_275',
}

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

    # Elo SD (team consistency)
    elo_sd_window: int = 10

    # Goal-based K-factor exponent: K_eff = K * (1 + |goal_diff|)^lambda
    # 0.0 = fixed K (backward compatible), >0 = big wins move ratings more
    elo_k_goal_lambda: float = 0.0

    # HT Elo parameters
    ht_elo_k_factor: float = 32.0
    ht_elo_home_advantage: float = 80.0

    # Pi-rating parameters
    pi_rating_lambda: float = 0.035
    pi_rating_gamma: float = 0.70
    pi_rating_c: float = 3.0

    # Niche market-specific EMA spans (independent per market)
    fouls_ema_span: int = 10
    cards_ema_span: int = 10
    shots_ema_span: int = 10
    corners_ema_span: int = 10

    # League-relative window for niche market engineers (EWM span)
    league_window: int = 50

    # League aggregate window (EWM span for league-level features)
    league_aggregate_window: int = 100

    # Referee feature windows
    referee_career_window: int = 30
    referee_recent_window: int = 10

    # Niche market window sizes
    fouls_window_sizes: List[int] = field(default_factory=lambda: [5, 10])
    cards_window_sizes: List[int] = field(default_factory=lambda: [5, 10])
    shots_window_sizes: List[int] = field(default_factory=lambda: [5, 10])
    corners_window_sizes: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Niche derived features (ratio + volatility)
    niche_volatility_window: int = 10
    niche_ratio_ema_span: int = 10

    # Dynamics features (distributional, momentum, regime, hurst, damped trend)
    dynamics_window: int = 10
    dynamics_short_ema: int = 5
    dynamics_long_ema: int = 15
    dynamics_long_window: int = 20
    dynamics_damping: float = 0.9
    dynamics_hurst_window: int = 15

    # Entropy features (permutation entropy, sample entropy)
    entropy_window: int = 15

    # Window ratio features (short/long EMA ratio)
    window_ratio_short_ema: int = 3
    window_ratio_long_ema: int = 12

    # Weighted Streak Index window
    wsi_window: int = 6
    wsi_weighting: str = "linear"

    # NegBin features for niche markets
    use_negbin_features: bool = True

    # Rolling z-score normalization (distribution shift mitigation)
    # Default False — enabled per-market via NICHE_NORMALIZE_MARKETS
    normalize_features: bool = False
    normalize_window: int = 0       # 0=expanding, >0=rolling window size
    normalize_min_periods: int = 30

    # League-relative features for niche markets
    # When True, niche market engineers add features relative to league norms
    use_league_relative: bool = True

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
                'sd_window': self.elo_sd_window,
                'k_goal_lambda': self.elo_k_goal_lambda,
            },
            'ht_elo': {
                'k_factor': self.ht_elo_k_factor,
                'home_advantage': self.ht_elo_home_advantage,
            },
            'pi_rating': {
                'lambda_': self.pi_rating_lambda,
                'gamma': self.pi_rating_gamma,
                'c': self.pi_rating_c,
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
                'use_league_relative': self.use_league_relative,
                'league_window': self.league_window,
            },
            'cards': {
                'window_sizes': self.cards_window_sizes,
                'ema_span': self.cards_ema_span,
                'use_league_relative': self.use_league_relative,
                'league_window': self.league_window,
            },
            'shots': {
                'window_sizes': self.shots_window_sizes,
                'ema_span': self.shots_ema_span,
                'use_league_relative': self.use_league_relative,
                'league_window': self.league_window,
            },
            'corners': {
                'window_sizes': self.corners_window_sizes,
                'ema_span': self.corners_ema_span,
                'use_league_relative': self.use_league_relative,
            },
            'referee': {
                'min_matches': 5,
                'recent_window': self.referee_recent_window,
                'career_window': self.referee_career_window,
            },
            'league_aggregate': {
                'min_matches': 20,
                'window': self.league_aggregate_window,
            },
            'streak': {
                'wsi_window': self.wsi_window,
                'wsi_weighting': self.wsi_weighting,
            },
            'niche_derived': {
                'volatility_window': self.niche_volatility_window,
                'ratio_ema_span': self.niche_ratio_ema_span,
            },
            'dynamics': {
                'window': self.dynamics_window,
                'short_ema': self.dynamics_short_ema,
                'long_ema': self.dynamics_long_ema,
                'long_window': self.dynamics_long_window,
                'damping_factor': self.dynamics_damping,
                'hurst_window': self.dynamics_hurst_window,
            },
            'entropy': {
                'window': self.entropy_window,
            },
            'window_ratio': {
                'short_ema': self.window_ratio_short_ema,
                'long_ema': self.window_ratio_long_ema,
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
            'elo_sd_window': self.elo_sd_window,
            'elo_k_goal_lambda': self.elo_k_goal_lambda,
            'ht_elo_k_factor': self.ht_elo_k_factor,
            'ht_elo_home_advantage': self.ht_elo_home_advantage,
            'form_window': self.form_window,
            'ema_span': self.ema_span,
            'poisson_lookback': self.poisson_lookback,
            'pi_rating_lambda': self.pi_rating_lambda,
            'pi_rating_gamma': self.pi_rating_gamma,
            'pi_rating_c': self.pi_rating_c,
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
            'niche_volatility_window': self.niche_volatility_window,
            'niche_ratio_ema_span': self.niche_ratio_ema_span,
            'dynamics_window': self.dynamics_window,
            'dynamics_short_ema': self.dynamics_short_ema,
            'dynamics_long_ema': self.dynamics_long_ema,
            'dynamics_long_window': self.dynamics_long_window,
            'dynamics_damping': self.dynamics_damping,
            'dynamics_hurst_window': self.dynamics_hurst_window,
            'entropy_window': self.entropy_window,
            'window_ratio_short_ema': self.window_ratio_short_ema,
            'window_ratio_long_ema': self.window_ratio_long_ema,
            'wsi_window': self.wsi_window,
            'wsi_weighting': self.wsi_weighting,
            'use_negbin_features': self.use_negbin_features,
            'normalize_features': self.normalize_features,
            'normalize_window': self.normalize_window,
            'normalize_min_periods': self.normalize_min_periods,
            'use_league_relative': self.use_league_relative,
            'league_window': self.league_window,
            'league_aggregate_window': self.league_aggregate_window,
            'referee_career_window': self.referee_career_window,
            'referee_recent_window': self.referee_recent_window,
        }
        params_str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:12]

    def save(self, path: Optional[Path] = None, params_dir: Optional[Path] = None) -> Path:
        """
        Save configuration to YAML file.

        Args:
            path: Optional path to save to. Defaults to {params_dir}/{bet_type}.yaml
            params_dir: Optional directory for feature params. Defaults to FEATURE_PARAMS_DIR.

        Returns:
            Path where config was saved
        """
        if path is None:
            output_dir = params_dir or FEATURE_PARAMS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{self.bet_type}.yaml"

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
    def load_for_bet_type(cls, bet_type: str, params_dir: Optional[Path] = None) -> 'BetTypeFeatureConfig':
        """
        Load configuration for a specific bet type.

        Looks for config in standard location: {params_dir}/{bet_type}.yaml
        Falls back to default.yaml if bet-type-specific config doesn't exist.
        Falls back to default values if no config files exist.

        Args:
            bet_type: The bet type to load config for
            params_dir: Optional directory for feature params. Defaults to FEATURE_PARAMS_DIR.

        Returns:
            BetTypeFeatureConfig instance
        """
        base_dir = params_dir or FEATURE_PARAMS_DIR

        # Try bet-type-specific config first
        bet_type_path = base_dir / f"{bet_type}.yaml"
        if bet_type_path.exists():
            config = cls.load(bet_type_path)
            # Enable normalization for niche markets if YAML doesn't specify
            if 'normalize_features' not in _yaml_keys(bet_type_path):
                if bet_type in NICHE_NORMALIZE_MARKETS:
                    config.normalize_features = True
            return config

        # Fall back to default config
        default_path = base_dir / "default.yaml"
        if default_path.exists():
            config = cls.load(default_path)
            config.bet_type = bet_type
            if bet_type in NICHE_NORMALIZE_MARKETS:
                config.normalize_features = True
            return config

        # Return default values — enable normalization for niche markets
        normalize = bet_type in NICHE_NORMALIZE_MARKETS
        return cls(bet_type=bet_type, normalize_features=normalize)

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
    # Bounds expanded based on R36/R37/R38 + R218-R220 boundary analysis
    'elo_k_factor': (5, 150, 'int'),           # ELO volatility (R50: under25@96, btts@99 hit ceiling at 100)
    'elo_home_advantage': (15, 350, 'int'),    # Home advantage points (R220: away_win@242 near 250 ceiling)
    'form_window': (1, 60, 'int'),             # Recent matches (R219: corners_o85@2 hit floor)
    'ema_span': (3, 35, 'int'),                # EMA smoothing (R218: fouls@19, cards@19 near 20 ceiling)
    'poisson_lookback': (5, 60, 'int'),        # Goal rate estimation (R39: under25@32, btts@29)

    # Elo SD window
    'elo_sd_window': (5, 20, 'int'),              # Rolling window for elo delta std

    # Goal-based K-factor exponent
    'elo_k_goal_lambda': (0.0, 0.5, 'float'),    # 0=fixed K, >0=big wins amplified

    # WSI window
    'wsi_window': (4, 10, 'int'),                 # Weighted Streak Index window

    # HT Elo parameters
    'ht_elo_k_factor': (10, 100, 'int'),
    'ht_elo_home_advantage': (20, 200, 'int'),

    # Pi-rating parameters
    'pi_rating_lambda': (0.01, 0.10, 'float'),   # Pi-rating learning rate
    'pi_rating_gamma': (0.30, 0.90, 'float'),    # Pi-rating cross-learning rate
    'pi_rating_c': (1.0, 5.0, 'float'),          # Pi-rating goal diff dampening

    # Phase 2: Extended parameters
    'half_life_days': (20.0, 150.0, 'float'),  # Time decay half-life
    'h2h_matches': (3, 12, 'int'),             # Head-to-head history
    'goal_diff_lookback': (1, 15, 'int'),      # Goal difference window (R218: cards@3 hit floor)
    'home_away_form_window': (1, 15, 'int'),   # Venue-specific form (R220: home_win@3 hit floor)

    # Niche market EMA spans
    'fouls_ema_span': (3, 35, 'int'),          # R218: fouls@19 near 20 ceiling
    'cards_ema_span': (2, 35, 'int'),          # R218: cards@19 near 20 ceiling
    'shots_ema_span': (3, 35, 'int'),
    'corners_ema_span': (3, 35, 'int'),        # R219: corners_o85@20 at ceiling

    # Niche derived features (ratio + volatility)
    'niche_volatility_window': (5, 25, 'int'),
    'niche_ratio_ema_span': (3, 25, 'int'),

    # Dynamics features (distributional, momentum, regime, hurst, damped trend)
    'dynamics_window': (5, 25, 'int'),
    'dynamics_short_ema': (3, 10, 'int'),
    'dynamics_long_ema': (10, 30, 'int'),
    'dynamics_hurst_window': (10, 25, 'int'),
    # Entropy features
    'entropy_window': (10, 25, 'int'),

    # Window ratio features (short/long EMA ratio)
    'window_ratio_short_ema': (2, 8, 'int'),
    'window_ratio_long_ema': (8, 20, 'int'),

    # League-relative window for niche market engineers
    'league_window': (20, 100, 'int'),

    # League aggregate window (all markets)
    'league_aggregate_window': (40, 200, 'int'),

    # Referee feature windows
    'referee_career_window': (15, 60, 'int'),
    'referee_recent_window': (5, 20, 'int'),
}

# Feature name regex → list of parameter names that affect the feature.
# Used by get_informed_search_space() to restrict Optuna search to params
# that actually matter for the deployed feature set.
FEATURE_TO_PARAMS_MAP: Dict[str, List[str]] = {
    # Direct engineer mappings
    r'(home_|away_)?elo|^elo_': ['elo_k_factor', 'elo_home_advantage', 'elo_sd_window', 'elo_k_goal_lambda'],
    r'^ht_elo': ['ht_elo_k_factor', 'ht_elo_home_advantage'],
    r'(home_|away_)?(pi_|bayes_)': ['pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    r'(form|streak|wsi|points_last|wins_last)': ['form_window'],
    r'poisson_|^expected_total|^expected_home|^expected_away|^glm_': ['poisson_lookback'],
    r'^h2h_': ['h2h_matches'],
    r'goal_diff|^gd_': ['goal_diff_lookback'],
    r'home_away_form|venue_gap': ['home_away_form_window'],
    r'^ref_|^referee_|_ref_': ['referee_career_window', 'referee_recent_window'],
    r'_vs_league|^league_': ['league_window', 'league_aggregate_window'],

    # Niche EMA (specific before generic)
    r'fouls.*_ema|fouls.*_momentum': ['fouls_ema_span', 'ema_span'],
    r'cards.*_ema|cards.*_momentum|yellows.*_ema': ['cards_ema_span', 'ema_span'],
    r'shots.*_ema|shots.*_momentum|sot.*_ema': ['shots_ema_span', 'ema_span'],
    r'corners.*_ema|corners.*_momentum': ['corners_ema_span', 'ema_span'],
    r'_ema$|_momentum$': ['ema_span'],

    # Dynamics
    r'_(kurtosis|skewness|damped_trend|hurst)': ['dynamics_window', 'dynamics_hurst_window'],
    r'_(first_diff|acceleration)': ['dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema'],
    r'_volatility': ['niche_volatility_window'],
    r'_ratio_ema': ['niche_ratio_ema_span'],
    r'_window_ratio': ['window_ratio_short_ema', 'window_ratio_long_ema'],
    r'_entropy|_pe_|_sampen': ['entropy_window'],

    # Cross-market TRANSITIVE deps (from cross_market.py _safe_get calls)
    r'fouls_int_cards': ['cards_ema_span', 'fouls_ema_span', 'shots_ema_span', 'poisson_lookback'],
    r'fouls_int_corners': ['corners_ema_span', 'cards_ema_span'],
    r'fouls_int_expected': ['poisson_lookback', 'cards_ema_span'],
    r'btts_int_|goals_int_': ['shots_ema_span', 'goal_diff_lookback'],
    r'corners_int_': ['shots_ema_span', 'fouls_ema_span'],
    r'shots_int_': ['corners_ema_span'],
    r'away_win_int_goaldiff_elo': ['elo_k_factor', 'elo_home_advantage'],
    r'away_win_int_(goaldiff_attack|xg)': ['poisson_lookback'],
    r'cross_shots': ['shots_ema_span'],
    r'cross_corners': ['corners_ema_span'],
    r'cross_fouls': ['fouls_ema_span'],
    r'cross_yellows': ['cards_ema_span'],

    # Parameterless (explicit documentation)
    r'^odds_': [],
    r'lineup_|formation_|coach_|squad_|xi_|missing_rating': [],
}


# Bet type categories and their primary parameters to optimize
BET_TYPE_PARAM_PRIORITIES = {
    # Match result markets
    'away_win': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
                 'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                 'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
                 'league_aggregate_window'],
    'home_win': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
                 'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                 'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
                 'league_aggregate_window'],
    'ah_minus_15': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
                    'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                    'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],

    # Goals markets
    'btts': ['elo_k_factor', 'elo_k_goal_lambda', 'form_window', 'ema_span', 'poisson_lookback',
             'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
             'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
             'league_aggregate_window'],
    'over25': ['elo_k_factor', 'elo_k_goal_lambda', 'form_window', 'ema_span', 'poisson_lookback',
               'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
               'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
               'league_aggregate_window'],
    'under25': ['elo_k_factor', 'elo_k_goal_lambda', 'form_window', 'ema_span', 'poisson_lookback',
                'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
                'league_aggregate_window'],

    # Niche markets (each gets its own market-specific EMA)
    'fouls': ['elo_k_factor', 'form_window', 'fouls_ema_span',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'niche_volatility_window', 'niche_ratio_ema_span',
              'dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema',
              'dynamics_hurst_window',
              'entropy_window',
              'window_ratio_short_ema', 'window_ratio_long_ema',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
              'league_window', 'league_aggregate_window',
              'referee_career_window', 'referee_recent_window'],
    'cards': ['elo_k_factor', 'form_window', 'cards_ema_span',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'niche_volatility_window', 'niche_ratio_ema_span',
              'dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema',
              'dynamics_hurst_window',
              'entropy_window',
              'window_ratio_short_ema', 'window_ratio_long_ema',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
              'league_window', 'league_aggregate_window',
              'referee_career_window', 'referee_recent_window'],
    'shots': ['elo_k_factor', 'form_window', 'shots_ema_span',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'niche_volatility_window', 'niche_ratio_ema_span',
              'dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema',
              'dynamics_hurst_window',
              'entropy_window',
              'window_ratio_short_ema', 'window_ratio_long_ema',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
              'league_window', 'league_aggregate_window'],
    'corners': ['elo_k_factor', 'form_window', 'corners_ema_span',
                'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                'niche_volatility_window', 'niche_ratio_ema_span',
                'dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema',
                'dynamics_hurst_window',
                'entropy_window',
                'window_ratio_short_ema', 'window_ratio_long_ema',
                'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
                'league_aggregate_window'],

    # HC/HT/H1 variant markets — alias to base market param priorities
    'cardshc': ['elo_k_factor', 'form_window', 'cards_ema_span',
                'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                'niche_volatility_window', 'niche_ratio_ema_span',
                'dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema',
                'dynamics_hurst_window',
                'entropy_window',
                'window_ratio_short_ema', 'window_ratio_long_ema',
                'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    'cornershc': ['elo_k_factor', 'form_window', 'corners_ema_span',
                  'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                  'niche_volatility_window', 'niche_ratio_ema_span',
                  'dynamics_window', 'dynamics_short_ema', 'dynamics_long_ema',
                  'dynamics_hurst_window',
                  'entropy_window',
                  'window_ratio_short_ema', 'window_ratio_long_ema',
                  'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    'ht': ['elo_k_factor', 'form_window', 'ema_span', 'poisson_lookback',
            'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
            'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
            'ht_elo_k_factor', 'ht_elo_home_advantage'],
    'home_win_h1': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
                     'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                     'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
                     'ht_elo_k_factor', 'ht_elo_home_advantage'],
    'away_win_h1': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
                     'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
                     'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c',
                     'ht_elo_k_factor', 'ht_elo_home_advantage'],
    'goals': ['elo_k_factor', 'elo_k_goal_lambda', 'form_window', 'ema_span', 'poisson_lookback',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    'hgoals': ['elo_k_factor', 'elo_k_goal_lambda', 'form_window', 'ema_span', 'poisson_lookback',
               'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
               'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    'agoals': ['elo_k_factor', 'elo_k_goal_lambda', 'form_window', 'ema_span', 'poisson_lookback',
               'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
               'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],

    # Double Chance markets
    'dc_1x': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    'dc_12': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
    'dc_x2': ['elo_k_factor', 'elo_home_advantage', 'elo_k_goal_lambda', 'form_window', 'ema_span',
              'half_life_days', 'h2h_matches', 'goal_diff_lookback', 'home_away_form_window',
              'pi_rating_lambda', 'pi_rating_gamma', 'pi_rating_c'],
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


def get_informed_search_space(
    bet_type: str,
    selected_features: List[str],
    min_params: int = 3,
) -> Dict[str, tuple]:
    """Get parameter search space restricted to params affecting selected features.

    Matches each feature name against FEATURE_TO_PARAMS_MAP patterns, unions
    the matched param names, and intersects with PARAMETER_SEARCH_SPACES.

    Args:
        bet_type: The bet type (for fallback to BET_TYPE_PARAM_PRIORITIES).
        selected_features: Feature names from the deployed model.
        min_params: Minimum params to include; pads from BET_TYPE_PARAM_PRIORITIES.

    Returns:
        Dict mapping parameter name to (min, max, type) tuples.
    """
    import logging
    logger = logging.getLogger(__name__)

    matched_params: set = set()
    for feat in selected_features:
        for pattern, params in FEATURE_TO_PARAMS_MAP.items():
            if re.search(pattern, feat):
                matched_params.update(params)

    # Intersect with valid search spaces
    informed = {p: PARAMETER_SEARCH_SPACES[p] for p in matched_params if p in PARAMETER_SEARCH_SPACES}

    # Pad to min_params from bet-type priorities if needed
    if len(informed) < min_params:
        priorities = BET_TYPE_PARAM_PRIORITIES.get(bet_type, ['elo_k_factor', 'form_window', 'ema_span'])
        for p in priorities:
            if p in PARAMETER_SEARCH_SPACES and p not in informed:
                informed[p] = PARAMETER_SEARCH_SPACES[p]
                if len(informed) >= min_params:
                    break

    # Fallback: if still empty (all parameterless features), use full bet-type space
    if not informed:
        logger.info(f"All features parameterless for {bet_type}, falling back to full search space")
        return get_search_space_for_bet_type(bet_type)

    full_size = len(get_search_space_for_bet_type(bet_type))
    logger.info(f"Informed search space: {len(informed)} params (from {full_size}) for {bet_type}")
    return informed


def load_selected_features_from_deployment(
    deployment_config_path: str,
    bet_type: str,
) -> Optional[List[str]]:
    """Load selected features for a bet type from deployment config.

    Handles both 'selected_features' and 'features' keys in the config.

    Args:
        deployment_config_path: Path to sniper_deployment.json.
        bet_type: The bet type to look up.

    Returns:
        List of feature names, or None if market not found.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        with open(deployment_config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load deployment config: {e}")
        return None

    markets = config.get('markets', {})
    market_cfg = markets.get(bet_type)
    if market_cfg is None:
        logger.info(f"Market {bet_type} not found in deployment config")
        return None

    features = market_cfg.get('selected_features') or market_cfg.get('features')
    if not features:
        logger.info(f"No features found for {bet_type} in deployment config")
        return None

    return features
