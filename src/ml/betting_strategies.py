"""
Betting Strategies using the Strategy Pattern

Each betting strategy encapsulates:
- Target variable creation (training)
- Strategy-specific feature engineering (training + inference)
- Evaluation logic (training)
- Bet recommendation generation (inference)

Adding a new bet type = adding a new Strategy class
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np


@dataclass
class StrategyConfig:
    """Configuration for a betting strategy."""
    enabled: bool = True
    approach: str = "classification"
    odds_column: Optional[str] = None
    probability_threshold: float = 0.5
    edge_threshold: float = 0.3
    expected_roi: float = 0.0
    p_profit: float = 0.0
    min_edge: float = 0.05
    max_stake_fraction: float = 0.05
    kelly_fraction: float = 0.25  # Fractional Kelly (0.25 = quarter Kelly, recommended)
    line_filter: Dict = field(default_factory=lambda: {'min': -4, 'max': -1.5})


class BettingStrategy(ABC):
    """
    Abstract base class for betting strategies.

    Each strategy defines how to:
    1. Create target variables from match data
    2. Engineer strategy-specific features
    3. Evaluate predictions and calculate ROI
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'asian_handicap', 'btts')."""
        pass

    @property
    @abstractmethod
    def is_regression(self) -> bool:
        """Whether this strategy uses regression (True) or classification (False)."""
        pass

    @property
    @abstractmethod
    def default_odds_column(self) -> str:
        """Default column name for odds."""
        pass

    @abstractmethod
    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Create target variable for this strategy.

        Args:
            df: Raw features DataFrame

        Returns:
            Tuple of (filtered_df, target_column_name)
        """
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add strategy-specific features.

        Default implementation returns df unchanged.
        Override in subclasses for custom features.
        """
        return df

    @abstractmethod
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray,
        df_test: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """
        Evaluate predictions for this strategy.

        Args:
            predictions: Dict of model_name -> predicted values
            y_test: Actual target values
            odds_test: Betting odds for test set
            df_test: Full test DataFrame (for strategies needing extra columns)

        Returns:
            List of result dicts sorted by ROI
        """
        pass

    @property
    @abstractmethod
    def bet_side(self) -> str:
        """Human-readable bet side (e.g., 'away', 'yes', 'over 2.5')."""
        pass

    def get_odds_column(self) -> str:
        """Get the odds column to use."""
        return self.config.odds_column or self.default_odds_column

    def create_recommendation(
        self,
        row: pd.Series,
        prediction: float,
        bankroll: float = 1000.0
    ) -> Optional[Dict[str, Any]]:
        """
        Create a bet recommendation for a single match.

        Args:
            row: Match data row with features and odds
            prediction: Model prediction (probability or margin)
            bankroll: Current bankroll for stake calculation

        Returns:
            Recommendation dict or None if bet doesn't meet criteria
        """
        odds_col = self.get_odds_column()
        odds = row.get(odds_col, 2.0)

        if pd.isna(odds) or odds <= 1.0:
            return None

        if self.is_regression:
            edge, prob = self._calc_regression_edge(row, prediction)
        else:
            prob = prediction
            implied_prob = 1 / odds
            edge = prob - implied_prob

        if edge < self.config.min_edge:
            return None

        # Expected value
        ev = (prob * odds) - 1

        # Fractional Kelly criterion
        # Full Kelly: f* = (p*b - q) / b = (p*odds - 1) / (odds - 1)
        # Fractional Kelly: f = fraction * f*
        # Using fractional Kelly (default 0.25) reduces variance and accounts for
        # probability estimation errors
        full_kelly = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
        kelly = self.config.kelly_fraction * full_kelly
        kelly = max(0, min(kelly, self.config.max_stake_fraction))

        return {
            'fixture_id': str(row.get('fixture_id', '')),
            'date': str(row.get('date', '')),
            'home_team': row.get('home_team_name', 'Home'),
            'away_team': row.get('away_team_name', 'Away'),
            'league': row.get('league', ''),
            'bet_type': self.name,
            'bet_side': self._format_bet_side(row),
            'odds': float(odds),
            'probability': float(prob),
            'edge': float(edge),
            'confidence': float(min(edge / 0.1, 1.0)),
            'expected_value': float(ev),
            'kelly_fraction': float(kelly),
            'recommended_stake': float(kelly * bankroll)
        }

    def _calc_regression_edge(self, row: pd.Series, prediction: float) -> Tuple[float, float]:
        """Calculate edge for regression strategies. Override in subclasses."""
        return 0.0, 0.5

    def _format_bet_side(self, row: pd.Series) -> str:
        """Format bet side string. Override for custom formatting."""
        return self.bet_side

    def calc_roi_bootstrap(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        odds: np.ndarray,
        n_boot: int = 1000
    ) -> Tuple[float, float, float, float]:
        """
        Calculate ROI with bootstrap confidence intervals.

        Returns:
            (mean_roi, ci_low, ci_high, p_profit)
        """
        if len(predictions) == 0 or predictions.sum() == 0:
            return 0, 0, 0, 0

        mask = predictions == 1
        wins = actuals[mask] == 1
        bet_odds = odds[mask]

        if len(wins) == 0:
            return 0, 0, 0, 0

        rois = []
        for _ in range(n_boot):
            idx = np.random.choice(len(wins), len(wins), replace=True)
            w, o = wins[idx], bet_odds[idx]
            profit = (w * (o - 1) - (~w) * 1).sum()
            rois.append(profit / len(w) * 100)

        return (
            np.mean(rois),
            np.percentile(rois, 2.5),
            np.percentile(rois, 97.5),
            (np.array(rois) > 0).mean()
        )


class AsianHandicapStrategy(BettingStrategy):
    """
    Asian Handicap betting strategy.

    Uses regression to predict goal margin and compares to bookmaker's line.
    Bets on away team when our predicted margin differs from bookie by threshold.
    """

    @property
    def name(self) -> str:
        return "asian_handicap"

    @property
    def is_regression(self) -> bool:
        return True

    @property
    def default_odds_column(self) -> str:
        return "avg_ah_away"

    @property
    def bet_side(self) -> str:
        return "away"

    def _calc_regression_edge(self, row: pd.Series, prediction: float) -> Tuple[float, float]:
        """Calculate edge for Asian Handicap."""
        ah_line = row.get('ah_line', 0)
        line_filter = self.config.line_filter

        if not (line_filter.get('min', -4) <= ah_line <= line_filter.get('max', -1.5)):
            return 0.0, 0.5

        bookie_margin = -ah_line
        edge = prediction - bookie_margin

        if edge < -self.config.edge_threshold:
            prob = 0.5 + abs(edge) * 0.2
            return abs(edge), prob

        return 0.0, 0.5

    def _format_bet_side(self, row: pd.Series) -> str:
        """Format bet side with handicap line."""
        ah_line = row.get('ah_line', 0)
        return f"away {ah_line:+.2f}"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create goal margin target for regression."""
        df_filtered = df[df['ah_line'].notna() & df['avg_ah_home'].notna()].copy()
        df_filtered['goal_margin'] = df_filtered['goal_difference']
        return df_filtered, 'goal_margin'

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Asian Handicap specific features."""
        df = df.copy()

        if 'home_xg_poisson' in df.columns and 'away_xg_poisson' in df.columns:
            df['xg_margin'] = df['home_xg_poisson'] - df['away_xg_poisson']

        if 'elo_diff' in df.columns:
            df['elo_expected_margin'] = df['elo_diff'] / 400

        if 'season_gd_diff' in df.columns and 'round_number' in df.columns:
            df['season_margin_per_game'] = df['season_gd_diff'] / (df['round_number'] + 1)

        if 'home_avg_goal_diff' in df.columns and 'away_avg_goal_diff' in df.columns:
            df['form_margin'] = df['home_avg_goal_diff'] - df['away_avg_goal_diff']

        margin_cols = ['elo_expected_margin', 'xg_margin', 'form_margin', 'season_margin_per_game']
        available = [c for c in margin_cols if c in df.columns]
        if available:
            df['composite_expected_margin'] = df[available].mean(axis=1)

        if 'ah_line' in df.columns and 'composite_expected_margin' in df.columns:
            df['bookie_expected_margin'] = -df['ah_line']
            df['margin_edge'] = df['composite_expected_margin'] - df['bookie_expected_margin']

        return df

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray,
        df_test: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """Evaluate using regression-based value betting."""
        if df_test is None:
            return []

        results = []
        ah_line = df_test['ah_line'].values
        bookie_margin = -ah_line
        odds_away = df_test['avg_ah_away'].values

        for model_name, pred in predictions.items():
            edge = pred - bookie_margin

            for thresh in [0.2, 0.3, 0.4, 0.5]:
                # Filter by line type (heavy favorites)
                line_mask = ah_line <= -1.5
                bet_mask = line_mask & (edge < -thresh)  # Away bets

                if bet_mask.sum() < 20:
                    continue

                wins = (y_test[bet_mask] + ah_line[bet_mask] < 0)
                bet_odds = odds_away[bet_mask]

                roi, ci_low, ci_high, p_profit = self.calc_roi_bootstrap(
                    np.ones(len(wins)), wins.astype(int), bet_odds
                )

                results.append({
                    'model': model_name,
                    'threshold': thresh,
                    'bets': int(bet_mask.sum()),
                    'win_rate': float(wins.mean()),
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

        results.sort(key=lambda x: x['roi'], reverse=True)
        return results


class BTTSStrategy(BettingStrategy):
    """
    Both Teams To Score (BTTS) strategy.

    Classification to predict if both teams will score.
    """

    @property
    def name(self) -> str:
        return "btts"

    @property
    def is_regression(self) -> bool:
        return False

    @property
    def default_odds_column(self) -> str:
        return "btts_yes_avg"

    @property
    def bet_side(self) -> str:
        return "yes"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create BTTS target."""
        df_filtered = df.copy()

        if 'btts' not in df_filtered.columns:
            # Calculate from goals if not present
            if 'home_goals' in df_filtered.columns and 'away_goals' in df_filtered.columns:
                # Filter rows with valid goals first
                valid_mask = df_filtered['home_goals'].notna() & df_filtered['away_goals'].notna()
                df_filtered = df_filtered[valid_mask].copy()
                df_filtered['btts'] = (
                    (df_filtered['home_goals'] > 0) & (df_filtered['away_goals'] > 0)
                ).astype(int)
            elif 'total_goals' in df_filtered.columns and 'goal_difference' in df_filtered.columns:
                # Filter rows with valid totals first
                valid_mask = df_filtered['total_goals'].notna() & df_filtered['goal_difference'].notna()
                df_filtered = df_filtered[valid_mask].copy()
                df_filtered['btts'] = (
                    (df_filtered['total_goals'] >= 2) &
                    (df_filtered['goal_difference'].abs() < df_filtered['total_goals'])
                ).astype(int)
            else:
                raise ValueError("Cannot create BTTS target: missing required columns (home_goals/away_goals or total_goals/goal_difference)")
        else:
            # Existing btts column - ensure it's numeric and filter NaN
            df_filtered['btts'] = pd.to_numeric(df_filtered['btts'], errors='coerce')

        # Filter out any remaining NaN targets
        df_filtered = df_filtered[df_filtered['btts'].notna()].copy()
        # Ensure integer type
        df_filtered['btts'] = df_filtered['btts'].astype(int)

        # Filter out NaN targets
        df_filtered = df_filtered[df_filtered['btts'].notna()].copy()

        return df_filtered, 'btts'

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BTTS specific features."""
        df = df.copy()

        # Scoring probabilities using Poisson
        if 'home_goals_scored_ema' in df.columns:
            df['home_scores_prob'] = 1 - np.exp(-df['home_goals_scored_ema'].fillna(1.5))
        if 'away_goals_scored_ema' in df.columns:
            df['away_scores_prob'] = 1 - np.exp(-df['away_goals_scored_ema'].fillna(1.2))

        # BTTS composite probability
        if 'home_scores_prob' in df.columns and 'away_scores_prob' in df.columns:
            df['btts_composite'] = df['home_scores_prob'] * df['away_scores_prob']

        # Attack/defense metrics
        if 'home_attack_strength' in df.columns and 'away_attack_strength' in df.columns:
            df['total_attack'] = df['home_attack_strength'] + df['away_attack_strength']
            df['min_attack'] = df[['home_attack_strength', 'away_attack_strength']].min(axis=1)

        if 'home_defense_strength' in df.columns and 'away_defense_strength' in df.columns:
            df['min_defense'] = df[['home_defense_strength', 'away_defense_strength']].min(axis=1)

        return df

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray,
        df_test: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """Evaluate BTTS predictions."""
        return self._evaluate_classification(predictions, y_test, odds_test)

    def _evaluate_classification(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray
    ) -> List[Dict]:
        """Standard classification evaluation."""
        results = []

        for model_name, proba in predictions.items():
            for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                bet_mask = proba >= thresh

                if bet_mask.sum() < 20:
                    continue

                if odds_test is not None:
                    roi, ci_low, ci_high, p_profit = self.calc_roi_bootstrap(
                        bet_mask.astype(int), y_test, odds_test
                    )
                else:
                    roi, ci_low, ci_high, p_profit = 0, 0, 0, 0

                results.append({
                    'model': model_name,
                    'threshold': thresh,
                    'bets': int(bet_mask.sum()),
                    'win_rate': float(y_test[bet_mask].mean()) if bet_mask.sum() > 0 else 0,
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

        results.sort(key=lambda x: x['roi'], reverse=True)
        return results


class MatchResultStrategy(BettingStrategy):
    """
    Base class for match result strategies (Home Win, Away Win, Draw).
    """

    @property
    def is_regression(self) -> bool:
        return False

    @property
    @abstractmethod
    def result_type(self) -> str:
        """'home_win', 'away_win', or 'draw'."""
        pass

    @property
    def bet_side(self) -> str:
        """Default implementation based on result_type."""
        return self.result_type.replace('_', ' ')

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Match result strategies use standard features."""
        return df

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray,
        df_test: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """Evaluate match result predictions."""
        results = []

        for model_name, proba in predictions.items():
            for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                bet_mask = proba >= thresh

                if bet_mask.sum() < 20:
                    continue

                if odds_test is not None:
                    roi, ci_low, ci_high, p_profit = self.calc_roi_bootstrap(
                        bet_mask.astype(int), y_test, odds_test
                    )
                else:
                    roi, ci_low, ci_high, p_profit = 0, 0, 0, 0

                results.append({
                    'model': model_name,
                    'threshold': thresh,
                    'bets': int(bet_mask.sum()),
                    'win_rate': float(y_test[bet_mask].mean()) if bet_mask.sum() > 0 else 0,
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

        results.sort(key=lambda x: x['roi'], reverse=True)
        return results


class AwayWinStrategy(MatchResultStrategy):
    """Away Win betting strategy."""

    @property
    def name(self) -> str:
        return "away_win"

    @property
    def result_type(self) -> str:
        return "away_win"

    @property
    def default_odds_column(self) -> str:
        return "avg_away_open"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create away win target."""
        df_filtered = df[df['avg_away_open'].notna()].copy()
        df_filtered['target'] = df_filtered['away_win'].astype(int)
        return df_filtered, 'target'


class HomeWinStrategy(MatchResultStrategy):
    """Home Win betting strategy."""

    @property
    def name(self) -> str:
        return "home_win"

    @property
    def result_type(self) -> str:
        return "home_win"

    @property
    def default_odds_column(self) -> str:
        return "avg_home_open"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create home win target."""
        df_filtered = df[df['avg_home_open'].notna()].copy()
        df_filtered['target'] = df_filtered['home_win'].astype(int)
        return df_filtered, 'target'


class TotalsStrategy(BettingStrategy):
    """Base class for Over/Under totals strategies."""

    @property
    def is_regression(self) -> bool:
        return False

    @property
    @abstractmethod
    def total_line(self) -> float:
        """The total line (e.g., 2.5)."""
        pass

    @property
    @abstractmethod
    def is_over(self) -> bool:
        """True for over, False for under."""
        pass

    @property
    def bet_side(self) -> str:
        """Format as 'over X.X' or 'under X.X'."""
        direction = "over" if self.is_over else "under"
        return f"{direction} {self.total_line}"

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Totals strategies use standard features."""
        return df

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray,
        df_test: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """Evaluate totals predictions."""
        results = []

        for model_name, proba in predictions.items():
            for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                bet_mask = proba >= thresh

                if bet_mask.sum() < 20:
                    continue

                if odds_test is not None:
                    roi, ci_low, ci_high, p_profit = self.calc_roi_bootstrap(
                        bet_mask.astype(int), y_test, odds_test
                    )
                else:
                    roi, ci_low, ci_high, p_profit = 0, 0, 0, 0

                results.append({
                    'model': model_name,
                    'threshold': thresh,
                    'bets': int(bet_mask.sum()),
                    'win_rate': float(y_test[bet_mask].mean()) if bet_mask.sum() > 0 else 0,
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

        results.sort(key=lambda x: x['roi'], reverse=True)
        return results


class Over25Strategy(TotalsStrategy):
    """Over 2.5 goals betting strategy."""

    @property
    def name(self) -> str:
        return "over25"

    @property
    def total_line(self) -> float:
        return 2.5

    @property
    def is_over(self) -> bool:
        return True

    @property
    def default_odds_column(self) -> str:
        return "avg_over25"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create over 2.5 target."""
        df_filtered = df[df['avg_over25'].notna()].copy()
        df_filtered['target'] = (df_filtered['total_goals'] > 2.5).astype(int)
        return df_filtered, 'target'


class Under25Strategy(TotalsStrategy):
    """Under 2.5 goals betting strategy."""

    @property
    def name(self) -> str:
        return "under25"

    @property
    def total_line(self) -> float:
        return 2.5

    @property
    def is_over(self) -> bool:
        return False

    @property
    def default_odds_column(self) -> str:
        return "avg_under25"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create under 2.5 target."""
        df_filtered = df[df['avg_under25'].notna()].copy()
        df_filtered['target'] = (df_filtered['total_goals'] <= 2.5).astype(int)
        return df_filtered, 'target'


# =============================================================================
# NICHE MARKET STRATEGIES
# =============================================================================

class NicheMarketStrategy(BettingStrategy):
    """Base class for niche market strategies (corners, cards, shots, fouls)."""

    def __init__(self, config: Optional[StrategyConfig] = None, line: Optional[float] = None):
        super().__init__(config)
        self._line = line  # None = use subclass default

    @property
    def line(self) -> float:
        return self._line if self._line is not None else self.default_line

    @property
    @abstractmethod
    def default_line(self) -> float:
        """Subclass default line (e.g., 4.5 for cards)."""
        pass

    @property
    def is_regression(self) -> bool:
        return False

    @property
    @abstractmethod
    def stat_column(self) -> str:
        """Column name for the stat (e.g., 'total_corners')."""
        pass

    @property
    @abstractmethod
    def ref_stat_column(self) -> str:
        """Column name for referee stat average."""
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add referee-based features if available."""
        return df

    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: np.ndarray,
        odds_test: np.ndarray,
        df_test: Optional[pd.DataFrame] = None
    ) -> List[Dict]:
        """Evaluate niche market predictions."""
        results = []

        for model_name, proba in predictions.items():
            for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                bet_mask = proba >= thresh

                if bet_mask.sum() < 10:
                    continue

                if odds_test is not None:
                    roi, ci_low, ci_high, p_profit = self.calc_roi_bootstrap(
                        bet_mask.astype(int), y_test, odds_test
                    )
                else:
                    roi, ci_low, ci_high, p_profit = 0, 0, 0, 0

                results.append({
                    'model': model_name,
                    'threshold': thresh,
                    'bets': int(bet_mask.sum()),
                    'win_rate': float(y_test[bet_mask].mean()) if bet_mask.sum() > 0 else 0,
                    'roi': float(roi),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high),
                    'p_profit': float(p_profit)
                })

        results.sort(key=lambda x: x['roi'], reverse=True)
        return results


class CornersStrategy(NicheMarketStrategy):
    """Corners betting strategy."""

    @property
    def default_line(self) -> float:
        return 9.5

    @property
    def name(self) -> str:
        if self._line is not None and self._line != self.default_line:
            return f"corners_over_{str(self.line).replace('.', '')}"
        return "corners"

    @property
    def stat_column(self) -> str:
        return "total_corners"

    @property
    def ref_stat_column(self) -> str:
        return "ref_corner_avg"

    @property
    def default_odds_column(self) -> str:
        line_str = str(self.line).replace('.', '_')
        return f"corners_over_{line_str}"

    @property
    def bet_side(self) -> str:
        return f"corners over {self.line}"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create corners target."""
        df_filtered = df.copy()
        if self.stat_column not in df_filtered.columns:
            raise ValueError(
                f"CornersStrategy requires '{self.stat_column}' column. "
                f"Available columns: {[c for c in df_filtered.columns if 'corner' in c.lower()]}"
            )
        # Filter out NaN stat values first
        df_filtered = df_filtered[df_filtered[self.stat_column].notna()].copy()
        df_filtered['target'] = (df_filtered[self.stat_column] > self.line).astype(int)
        return df_filtered, 'target'


class CardsStrategy(NicheMarketStrategy):
    """Cards betting strategy."""

    @property
    def default_line(self) -> float:
        return 4.5

    @property
    def name(self) -> str:
        if self._line is not None and self._line != self.default_line:
            return f"cards_over_{str(self.line).replace('.', '')}"
        return "cards"

    @property
    def stat_column(self) -> str:
        return "total_cards"

    @property
    def ref_stat_column(self) -> str:
        return "ref_cards_avg"

    @property
    def default_odds_column(self) -> str:
        line_str = str(self.line).replace('.', '_')
        return f"cards_over_{line_str}"

    @property
    def bet_side(self) -> str:
        return f"cards over {self.line}"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create cards target."""
        df_filtered = df.copy()
        if self.stat_column not in df_filtered.columns:
            raise ValueError(
                f"CardsStrategy requires '{self.stat_column}' column. "
                f"Available columns: {[c for c in df_filtered.columns if 'card' in c.lower()]}"
            )
        df_filtered = df_filtered[df_filtered[self.stat_column].notna()].copy()
        df_filtered['target'] = (df_filtered[self.stat_column] > self.line).astype(int)
        return df_filtered, 'target'


class ShotsStrategy(NicheMarketStrategy):
    """Shots betting strategy."""

    @property
    def default_line(self) -> float:
        return 24.5

    @property
    def name(self) -> str:
        if self._line is not None and self._line != self.default_line:
            return f"shots_over_{str(self.line).replace('.', '')}"
        return "shots"

    @property
    def stat_column(self) -> str:
        return "total_shots"

    @property
    def ref_stat_column(self) -> str:
        return "ref_shots_avg"

    @property
    def default_odds_column(self) -> str:
        line_str = str(self.line).replace('.', '_')
        return f"shots_over_{line_str}"

    @property
    def bet_side(self) -> str:
        return f"shots over {self.line}"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create shots target."""
        df_filtered = df.copy()
        if self.stat_column not in df_filtered.columns:
            raise ValueError(
                f"ShotsStrategy requires '{self.stat_column}' column. "
                f"Available columns: {[c for c in df_filtered.columns if 'shot' in c.lower()]}"
            )
        df_filtered = df_filtered[df_filtered[self.stat_column].notna()].copy()
        df_filtered['target'] = (df_filtered[self.stat_column] > self.line).astype(int)
        return df_filtered, 'target'


class FoulsStrategy(NicheMarketStrategy):
    """Fouls betting strategy."""

    @property
    def default_line(self) -> float:
        return 24.5

    @property
    def name(self) -> str:
        if self._line is not None and self._line != self.default_line:
            return f"fouls_over_{str(self.line).replace('.', '')}"
        return "fouls"

    @property
    def stat_column(self) -> str:
        return "total_fouls"

    @property
    def ref_stat_column(self) -> str:
        return "ref_fouls_avg"

    @property
    def default_odds_column(self) -> str:
        line_str = str(self.line).replace('.', '_')
        return f"fouls_over_{line_str}"

    @property
    def bet_side(self) -> str:
        return f"fouls over {self.line}"

    def create_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Create fouls target."""
        df_filtered = df.copy()
        if self.stat_column not in df_filtered.columns:
            raise ValueError(
                f"FoulsStrategy requires '{self.stat_column}' column. "
                f"Available columns: {[c for c in df_filtered.columns if 'foul' in c.lower()]}"
            )
        df_filtered = df_filtered[df_filtered[self.stat_column].notna()].copy()
        df_filtered['target'] = (df_filtered[self.stat_column] > self.line).astype(int)
        return df_filtered, 'target'


# Strategy Registry
STRATEGY_REGISTRY: Dict[str, type] = {
    # Main markets
    'asian_handicap': AsianHandicapStrategy,
    'btts': BTTSStrategy,
    'away_win': AwayWinStrategy,
    'home_win': HomeWinStrategy,
    'over25': Over25Strategy,
    'under25': Under25Strategy,
    # Niche markets (base)
    'corners': CornersStrategy,
    'cards': CardsStrategy,
    'shots': ShotsStrategy,
    'fouls': FoulsStrategy,
    # Niche market line variants
    'cards_over_35': CardsStrategy,
    'cards_over_55': CardsStrategy,
    'cards_over_65': CardsStrategy,
    'corners_over_85': CornersStrategy,
    'corners_over_105': CornersStrategy,
    'corners_over_115': CornersStrategy,
    'shots_over_225': ShotsStrategy,
    'shots_over_265': ShotsStrategy,
    'shots_over_285': ShotsStrategy,
    'fouls_over_225': FoulsStrategy,
    'fouls_over_265': FoulsStrategy,
    'fouls_over_285': FoulsStrategy,
}

# Line lookup for niche market variants
NICHE_LINE_LOOKUP: Dict[str, float] = {
    'cards_over_35': 3.5,
    'cards_over_55': 5.5,
    'cards_over_65': 6.5,
    'corners_over_85': 8.5,
    'corners_over_105': 10.5,
    'corners_over_115': 11.5,
    'shots_over_225': 22.5,
    'shots_over_265': 26.5,
    'shots_over_285': 28.5,
    'fouls_over_225': 22.5,
    'fouls_over_265': 26.5,
    'fouls_over_285': 28.5,
}

# Maps line variants to their base market for feature params sharing
BASE_MARKET_MAP: Dict[str, str] = {
    'cards_over_35': 'cards',
    'cards_over_55': 'cards',
    'cards_over_65': 'cards',
    'corners_over_85': 'corners',
    'corners_over_105': 'corners',
    'corners_over_115': 'corners',
    'shots_over_225': 'shots',
    'shots_over_265': 'shots',
    'shots_over_285': 'shots',
    'fouls_over_225': 'fouls',
    'fouls_over_265': 'fouls',
    'fouls_over_285': 'fouls',
}


def get_strategy(name: str, config: Optional[StrategyConfig] = None) -> BettingStrategy:
    """
    Factory function to get a strategy by name.

    Args:
        name: Strategy name (e.g., 'asian_handicap', 'btts', 'cards_over_35')
        config: Optional configuration

    Returns:
        BettingStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    strategy_class = STRATEGY_REGISTRY[name]
    line = NICHE_LINE_LOOKUP.get(name)
    if line is not None:
        return strategy_class(config, line=line)
    return strategy_class(config)


def register_strategy(name: str, strategy_class: type) -> None:
    """
    Register a new strategy class.

    Args:
        name: Strategy identifier
        strategy_class: Strategy class (must inherit from BettingStrategy)
    """
    if not issubclass(strategy_class, BettingStrategy):
        raise TypeError(f"{strategy_class} must inherit from BettingStrategy")

    STRATEGY_REGISTRY[name] = strategy_class
