"""
Feature Engineer Registry

Registry pattern for feature engineers:
- Centralized registration of all feature engineers
- Configuration-driven instantiation
- Easy to add new engineers without modifying pipeline code
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Callable
import logging

import pandas as pd

from src.features.interfaces import IFeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineerConfig:
    """Configuration for a feature engineer."""
    name: str
    enabled: bool = True
    required: bool = False  # If True, pipeline fails if engineer fails
    requires_data: List[str] = field(default_factory=lambda: ['matches'])
    params: Dict[str, Any] = field(default_factory=dict)


class FeatureEngineerRegistry:
    """
    Registry for feature engineers.

    Usage:
        # Register engineers
        registry = FeatureEngineerRegistry()
        registry.register('team_form', TeamFormFeatureEngineer)
        registry.register('elo', ELORatingFeatureEngineer, default_params={'k_factor': 32})

        # Or use auto-discovery
        registry.auto_register()

        # Get configured engineers
        engineers = registry.get_engineers(config)

        # Create features
        features = registry.create_all_features(data, config)
    """

    def __init__(self):
        self._registry: Dict[str, Type[IFeatureEngineer]] = {}
        self._default_params: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        engineer_class: Type[IFeatureEngineer],
        default_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a feature engineer class.

        Args:
            name: Unique identifier for the engineer
            engineer_class: The engineer class to register
            default_params: Default parameters for instantiation
        """
        self._registry[name] = engineer_class
        if default_params:
            self._default_params[name] = default_params

    def unregister(self, name: str) -> None:
        """Remove an engineer from the registry."""
        self._registry.pop(name, None)
        self._default_params.pop(name, None)

    def get(self, name: str, **params) -> IFeatureEngineer:
        """
        Get an instance of a registered engineer.

        Args:
            name: Engineer name
            **params: Override default parameters

        Returns:
            Instantiated feature engineer

        Raises:
            KeyError: If engineer not found
        """
        if name not in self._registry:
            raise KeyError(f"Unknown feature engineer: {name}. Available: {list(self._registry.keys())}")

        engineer_class = self._registry[name]
        merged_params = {**self._default_params.get(name, {}), **params}

        return engineer_class(**merged_params) if merged_params else engineer_class()

    def list_engineers(self) -> List[str]:
        """Get list of registered engineer names."""
        return list(self._registry.keys())

    def get_engineers(
        self,
        configs: List[FeatureEngineerConfig]
    ) -> List[tuple[str, IFeatureEngineer]]:
        """
        Get list of configured engineers.

        Args:
            configs: List of engineer configurations

        Returns:
            List of (name, engineer) tuples for enabled engineers
        """
        engineers = []
        for cfg in configs:
            if not cfg.enabled:
                continue

            if cfg.name not in self._registry:
                logger.warning(f"Engineer '{cfg.name}' not registered, skipping")
                continue

            try:
                engineer = self.get(cfg.name, **cfg.params)
                engineers.append((cfg.name, engineer))
            except Exception as e:
                logger.error(f"Failed to instantiate '{cfg.name}': {e}")
                if cfg.required:
                    raise

        return engineers

    def create_all_features(
        self,
        data: Dict[str, pd.DataFrame],
        configs: List[FeatureEngineerConfig],
        on_error: str = 'warn'  # 'warn', 'raise', 'skip'
    ) -> List[pd.DataFrame]:
        """
        Create features using all configured engineers.

        Args:
            data: Dict of DataFrames (matches, player_stats, etc.)
            configs: List of engineer configurations
            on_error: Error handling ('warn', 'raise', 'skip')

        Returns:
            List of feature DataFrames
        """
        feature_dfs = []

        for cfg in configs:
            if not cfg.enabled:
                continue

            # Check required data
            missing_data = [d for d in cfg.requires_data if d not in data]
            if missing_data:
                logger.debug(f"Skipping '{cfg.name}': missing data {missing_data}")
                continue

            try:
                logger.info(f"Creating {cfg.name} features...")
                engineer = self.get(cfg.name, **cfg.params)
                features = engineer.create_features(data)

                if features is not None and not features.empty and len(features.columns) > 1:
                    feature_dfs.append(features)
                    logger.debug(f"  Created {len(features.columns)} features")

            except Exception as e:
                if cfg.required or on_error == 'raise':
                    raise
                elif on_error == 'warn':
                    logger.warning(f"Could not create {cfg.name} features: {e}")
                # else: skip silently

        return feature_dfs


# Global registry instance
_global_registry = None


def get_registry() -> FeatureEngineerRegistry:
    """Get the global feature engineer registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FeatureEngineerRegistry()
        _register_all_engineers(_global_registry)
    return _global_registry


def _register_all_engineers(registry: FeatureEngineerRegistry) -> None:
    """Register all known feature engineers."""
    from src.features.engineers import (
        TeamFormFeatureEngineer,
        TeamStatsFeatureEngineer,
        MatchOutcomeFeatureEngineer,
        HeadToHeadFeatureEngineer,
        ExponentialMovingAverageFeatureEngineer,
        ELORatingFeatureEngineer,
        PoissonFeatureEngineer,
        GoalDifferenceFeatureEngineer,
        HomeAwayFormFeatureEngineer,
        RestDaysFeatureEngineer,
        LeaguePositionFeatureEngineer,
        StreakFeatureEngineer,
        FormationFeatureEngineer,
        CoachFeatureEngineer,
        LineupStabilityFeatureEngineer,
        StarPlayerFeatureEngineer,
        TeamRatingFeatureEngineer,
        KeyPlayerAbsenceFeatureEngineer,
        DisciplineFeatureEngineer,
        GoalTimingFeatureEngineer,
        SeasonPhaseFeatureEngineer,
        DerbyFeatureEngineer,
        MatchImportanceFeatureEngineer,
        RefereeFeatureEngineer,
        CornerFeatureEngineer,
        # Niche markets
        FoulsFeatureEngineer,
        CardsFeatureEngineer,
        ShotsFeatureEngineer,
        # External
        WeatherFeatureEngineer,
        # New Phase 3 engineers
        FixtureCongestionEngineer,
        CLVDiagnosticEngineer,
        CLVOutcomeFeatureEngineer,
    )
    from src.features.engineers.cross_market import CrossMarketFeatureEngineer

    # Core features (always included)
    registry.register('team_form', TeamFormFeatureEngineer, {'n_matches': 5})
    registry.register('ema', ExponentialMovingAverageFeatureEngineer, {'span': 5})
    registry.register('elo', ELORatingFeatureEngineer, {'k_factor': 32.0, 'home_advantage': 100.0})
    registry.register('poisson', PoissonFeatureEngineer, {'lookback_matches': 10})
    registry.register('goal_diff', GoalDifferenceFeatureEngineer, {'lookback_matches': 5})
    registry.register('outcome', MatchOutcomeFeatureEngineer)

    # Form features
    registry.register('h2h', HeadToHeadFeatureEngineer, {'n_h2h': 5})
    registry.register('home_away_form', HomeAwayFormFeatureEngineer, {'n_matches': 5})
    registry.register('rest_days', RestDaysFeatureEngineer)
    registry.register('league_position', LeaguePositionFeatureEngineer)
    registry.register('streak', StreakFeatureEngineer)

    # V4 features (require additional data)
    registry.register('team_stats', TeamStatsFeatureEngineer, {'span': 5})
    registry.register('formation', FormationFeatureEngineer)
    registry.register('coach', CoachFeatureEngineer, {'lookback_matches': 5})
    registry.register('lineup_stability', LineupStabilityFeatureEngineer, {'lookback_matches': 5})
    registry.register('star_player', StarPlayerFeatureEngineer, {'top_n': 3, 'min_matches': 5})
    registry.register('team_rating', TeamRatingFeatureEngineer, {'lookback_matches': 10})
    registry.register('key_player_absence', KeyPlayerAbsenceFeatureEngineer, {'top_n': 5, 'lookback_matches': 10})
    registry.register('discipline', DisciplineFeatureEngineer, {'lookback_matches': 5})
    registry.register('goal_timing', GoalTimingFeatureEngineer, {'lookback_matches': 10})

    # Context features
    registry.register('season_phase', SeasonPhaseFeatureEngineer)
    registry.register('derby', DerbyFeatureEngineer)
    registry.register('match_importance', MatchImportanceFeatureEngineer)
    registry.register('referee', RefereeFeatureEngineer, {'min_matches': 5})

    # Niche market features
    registry.register('corners', CornerFeatureEngineer, {
        'window_sizes': [5, 10],
        'min_matches': 3,
        'use_ema': True,
        'ema_span': 10
    })
    registry.register('fouls', FoulsFeatureEngineer, {
        'window_sizes': [5, 10],
        'min_matches': 3,
        'ema_span': 10
    })
    registry.register('cards', CardsFeatureEngineer, {
        'window_sizes': [5, 10],
        'min_matches': 3,
        'ema_span': 10
    })
    registry.register('shots', ShotsFeatureEngineer, {
        'window_sizes': [5, 10],
        'min_matches': 3,
        'ema_span': 10
    })

    # External factors
    registry.register('weather', WeatherFeatureEngineer)

    # Cross-market interaction features
    registry.register('cross_market', CrossMarketFeatureEngineer)

    # New Phase 3 features: Fixture congestion and CLV diagnostics
    registry.register('fixture_congestion', FixtureCongestionEngineer, {
        'past_window_days': 14,
        'future_window_days': 14,
        'high_congestion_threshold': 4,
    })
    registry.register('clv_diagnostic', CLVDiagnosticEngineer, {
        'lookback_matches': 20,
        'min_matches': 5,
        'ema_span': 10,
    })
    registry.register('clv_outcome', CLVOutcomeFeatureEngineer, {
        'lookback_matches': 30,
        'min_matches': 10,
    })


# Default feature configurations
DEFAULT_FEATURE_CONFIGS = [
    # Core features (always enabled, required)
    FeatureEngineerConfig('team_form', enabled=True, required=True),
    FeatureEngineerConfig('ema', enabled=True, required=True),
    FeatureEngineerConfig('elo', enabled=True, required=True),
    FeatureEngineerConfig('poisson', enabled=True, required=True),
    FeatureEngineerConfig('goal_diff', enabled=True, required=True),

    # Form features
    FeatureEngineerConfig('h2h', enabled=True),
    FeatureEngineerConfig('home_away_form', enabled=True),
    FeatureEngineerConfig('rest_days', enabled=True),
    FeatureEngineerConfig('league_position', enabled=True),
    FeatureEngineerConfig('streak', enabled=True),

    # V4 features (optional, require additional data)
    FeatureEngineerConfig('team_stats', enabled=True, requires_data=['matches', 'player_stats']),
    FeatureEngineerConfig('formation', enabled=True, requires_data=['matches', 'lineups']),
    FeatureEngineerConfig('coach', enabled=True, requires_data=['matches', 'lineups']),
    FeatureEngineerConfig('lineup_stability', enabled=True, requires_data=['matches', 'lineups']),
    FeatureEngineerConfig('star_player', enabled=True, requires_data=['matches', 'player_stats']),
    FeatureEngineerConfig('team_rating', enabled=True, requires_data=['matches', 'player_stats']),
    FeatureEngineerConfig('key_player_absence', enabled=True, requires_data=['matches', 'player_stats', 'lineups']),
    FeatureEngineerConfig('discipline', enabled=True, requires_data=['matches', 'events']),
    FeatureEngineerConfig('goal_timing', enabled=True, requires_data=['matches', 'events']),

    # Context features
    FeatureEngineerConfig('season_phase', enabled=True),
    FeatureEngineerConfig('derby', enabled=True),
    FeatureEngineerConfig('match_importance', enabled=True),
    FeatureEngineerConfig('referee', enabled=True),

    # Niche market features
    FeatureEngineerConfig('corners', enabled=True, requires_data=['matches', 'match_stats']),
    FeatureEngineerConfig('fouls', enabled=True, requires_data=['matches', 'match_stats']),
    FeatureEngineerConfig('cards', enabled=True, requires_data=['matches', 'match_stats']),
    FeatureEngineerConfig('shots', enabled=True, requires_data=['matches', 'match_stats']),

    # External factors
    FeatureEngineerConfig('weather', enabled=True),

    # Cross-market interaction features
    FeatureEngineerConfig('cross_market', enabled=True),

    # New Phase 3 features: Fixture congestion and CLV diagnostics
    FeatureEngineerConfig('fixture_congestion', enabled=True),
    FeatureEngineerConfig('clv_diagnostic', enabled=True, requires_data=['matches']),
    FeatureEngineerConfig('clv_outcome', enabled=True, requires_data=['matches']),

    # Target (always last, required)
    FeatureEngineerConfig('outcome', enabled=True, required=True),
]


def get_default_configs(config=None) -> List[FeatureEngineerConfig]:
    """
    Get default feature configurations, optionally customized from Config.

    Args:
        config: Optional Config object to customize parameters

    Returns:
        List of FeatureEngineerConfig
    """
    configs = []

    for default_cfg in DEFAULT_FEATURE_CONFIGS:
        cfg = FeatureEngineerConfig(
            name=default_cfg.name,
            enabled=default_cfg.enabled,
            required=default_cfg.required,
            requires_data=default_cfg.requires_data.copy(),
            params=default_cfg.params.copy()
        )

        # Customize from Config if provided
        if config is not None:
            features_config = getattr(config, 'features', None)
            if features_config:
                # Map config attributes to engineer params
                if cfg.name == 'team_form' and hasattr(features_config, 'form_window'):
                    cfg.params['n_matches'] = features_config.form_window
                elif cfg.name == 'ema' and hasattr(features_config, 'ema_span'):
                    cfg.params['span'] = features_config.ema_span
                elif cfg.name == 'h2h' and not getattr(features_config, 'include_h2h', True):
                    cfg.enabled = False
                elif cfg.name == 'team_stats' and not getattr(features_config, 'include_team_stats', True):
                    cfg.enabled = False
                elif cfg.name == 'lineup_stability' and hasattr(features_config, 'lineup_lookback'):
                    cfg.params['lookback_matches'] = features_config.lineup_lookback
                elif cfg.name == 'star_player':
                    if hasattr(features_config, 'star_top_n'):
                        cfg.params['top_n'] = features_config.star_top_n
                    if hasattr(features_config, 'star_min_matches'):
                        cfg.params['min_matches'] = features_config.star_min_matches
                elif cfg.name in ['team_rating', 'key_player_absence'] and hasattr(features_config, 'rating_lookback'):
                    cfg.params['lookback_matches'] = features_config.rating_lookback
                elif cfg.name == 'discipline' and hasattr(features_config, 'discipline_lookback'):
                    cfg.params['lookback_matches'] = features_config.discipline_lookback
                elif cfg.name == 'goal_timing' and hasattr(features_config, 'goal_timing_lookback'):
                    cfg.params['lookback_matches'] = features_config.goal_timing_lookback

        configs.append(cfg)

    return configs


def create_configs_with_bet_type_params(
    bet_type_config: 'BetTypeFeatureConfig',
    base_config=None,
) -> List[FeatureEngineerConfig]:
    """
    Create feature configurations with bet-type-specific parameters.

    This function allows overriding feature engineer parameters based on
    a BetTypeFeatureConfig, which can contain optimized parameters for
    specific bet types (e.g., different ELO k-factors for away_win vs fouls).

    Args:
        bet_type_config: BetTypeFeatureConfig with custom parameters
        base_config: Optional Config object for additional customization

    Returns:
        List of FeatureEngineerConfig with params applied

    Example:
        from src.features.config_manager import BetTypeFeatureConfig

        # Create custom config
        config = BetTypeFeatureConfig(
            bet_type='away_win',
            elo_k_factor=40,
            form_window=7,
        )

        # Get configs with custom params
        configs = create_configs_with_bet_type_params(config)
    """
    # Import here to avoid circular import
    from src.features.config_manager import BetTypeFeatureConfig

    # Start with default configs (optionally modified by base_config)
    configs = get_default_configs(base_config)

    # Get registry params from bet_type_config
    registry_params = bet_type_config.to_registry_params()

    # Override params for each engineer
    for cfg in configs:
        if cfg.name in registry_params:
            cfg.params.update(registry_params[cfg.name])

    return configs


def get_registry_with_params(bet_type_config: 'BetTypeFeatureConfig') -> 'FeatureEngineerRegistry':
    """
    Get a registry instance configured with bet-type-specific parameters.

    This is a convenience function that returns a new registry instance
    with the default params updated based on BetTypeFeatureConfig.

    Note: This modifies the default params on a new registry instance,
    so it won't affect the global registry.

    Args:
        bet_type_config: BetTypeFeatureConfig with custom parameters

    Returns:
        FeatureEngineerRegistry with modified default params
    """
    # Import here to avoid circular import
    from src.features.config_manager import BetTypeFeatureConfig

    # Create new registry
    registry = FeatureEngineerRegistry()
    _register_all_engineers(registry)

    # Get custom params
    registry_params = bet_type_config.to_registry_params()

    # Update default params
    for engineer_name, params in registry_params.items():
        if engineer_name in registry._default_params:
            registry._default_params[engineer_name].update(params)
        else:
            registry._default_params[engineer_name] = params

    return registry
