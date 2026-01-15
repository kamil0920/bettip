"""
Feature engineering module.

This package contains all feature engineers organized by category:
- base: Base class for all engineers
- form: Team form and streak features
- stats: Team statistics and goal features
- ratings: ELO ratings and Poisson-based features
- h2h: Head-to-head and derby features
- context: Match context features (rest days, position, importance)
- lineup: Lineup and player-related features
- external: External factors (referee, weather)
"""

from src.features.engineers.base import BaseFeatureEngineer

from src.features.engineers.form import (
    TeamFormFeatureEngineer,
    ExponentialMovingAverageFeatureEngineer,
    HomeAwayFormFeatureEngineer,
    StreakFeatureEngineer,
)

from src.features.engineers.stats import (
    TeamStatsFeatureEngineer,
    GoalDifferenceFeatureEngineer,
    GoalTimingFeatureEngineer,
    DisciplineFeatureEngineer,
)

from src.features.engineers.ratings import (
    ELORatingFeatureEngineer,
    PoissonFeatureEngineer,
    TeamRatingFeatureEngineer,
)

from src.features.engineers.h2h import (
    HeadToHeadFeatureEngineer,
    DerbyFeatureEngineer,
)

from src.features.engineers.context import (
    MatchOutcomeFeatureEngineer,
    RestDaysFeatureEngineer,
    LeaguePositionFeatureEngineer,
    SeasonPhaseFeatureEngineer,
    MatchImportanceFeatureEngineer,
)

from src.features.engineers.lineup import (
    FormationFeatureEngineer,
    CoachFeatureEngineer,
    LineupStabilityFeatureEngineer,
    StarPlayerFeatureEngineer,
    KeyPlayerAbsenceFeatureEngineer,
)

from src.features.engineers.external import (
    RefereeFeatureEngineer,
    WeatherFeatureEngineer,
)

__all__ = [
    # Base
    "BaseFeatureEngineer",
    # Form
    "TeamFormFeatureEngineer",
    "ExponentialMovingAverageFeatureEngineer",
    "HomeAwayFormFeatureEngineer",
    "StreakFeatureEngineer",
    # Stats
    "TeamStatsFeatureEngineer",
    "GoalDifferenceFeatureEngineer",
    "GoalTimingFeatureEngineer",
    "DisciplineFeatureEngineer",
    # Ratings
    "ELORatingFeatureEngineer",
    "PoissonFeatureEngineer",
    "TeamRatingFeatureEngineer",
    # H2H
    "HeadToHeadFeatureEngineer",
    "DerbyFeatureEngineer",
    # Context
    "MatchOutcomeFeatureEngineer",
    "RestDaysFeatureEngineer",
    "LeaguePositionFeatureEngineer",
    "SeasonPhaseFeatureEngineer",
    "MatchImportanceFeatureEngineer",
    # Lineup
    "FormationFeatureEngineer",
    "CoachFeatureEngineer",
    "LineupStabilityFeatureEngineer",
    "StarPlayerFeatureEngineer",
    "KeyPlayerAbsenceFeatureEngineer",
    # External
    "RefereeFeatureEngineer",
    "WeatherFeatureEngineer",
]
