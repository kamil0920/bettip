"""Feature engineering - ELO ratings and Poisson-based features."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class ELORatingFeatureEngineer(BaseFeatureEngineer):
    """
    Creates ELO rating features for teams.

    ELO is a rating system that updates after each match based on:
    - Expected outcome vs actual outcome
    - K-factor (sensitivity to recent results)
    - Home advantage adjustment

    Standard ELO formula:
    R_new = R_old + K * (S - E)
    where:
    - S = actual score (1 for win, 0.5 for draw, 0 for loss)
    - E = expected score = 1 / (1 + 10^((R_opponent - R_self) / 400))
    """

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
    ):
        """
        Args:
            initial_rating: Starting ELO for new teams
            k_factor: How much ratings change per match (higher = more volatile)
            home_advantage: ELO points added for home team in expected calc
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate ELO ratings for each team before each match.

        Maintains three rating tracks per team:
        - Overall ELO (updated every match)
        - Home ELO (updated only on home matches)
        - Away ELO (updated only on away matches)

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with ELO features
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Initialize ratings
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())
        team_ratings = {team_id: self.initial_rating for team_id in all_teams}
        # Venue-specific ratings
        home_venue_ratings = {team_id: self.initial_rating for team_id in all_teams}
        away_venue_ratings = {team_id: self.initial_rating for team_id in all_teams}

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            home_elo = team_ratings[home_id]
            away_elo = team_ratings[away_id]

            home_expected = self._expected_score(home_elo + self.home_advantage, away_elo)
            away_expected = 1 - home_expected

            elo_diff = home_elo - away_elo

            # Venue-specific ratings (before this match)
            home_venue_elo = home_venue_ratings[home_id]
            away_venue_elo = away_venue_ratings[away_id]
            venue_elo_diff = home_venue_elo - away_venue_elo

            # Venue dependency: how much better/worse a team is at home vs away
            home_team_venue_gap = home_venue_ratings[home_id] - away_venue_ratings[home_id]
            away_team_venue_gap = home_venue_ratings[away_id] - away_venue_ratings[away_id]

            features = {
                'fixture_id': match['fixture_id'],
                'home_elo': home_elo,
                'away_elo': away_elo,
                'elo_diff': elo_diff,
                'home_win_prob_elo': home_expected,
                'away_win_prob_elo': away_expected,
                # Venue-specific features
                'home_venue_elo': home_venue_elo,
                'away_venue_elo': away_venue_elo,
                'venue_elo_diff': venue_elo_diff,
                'home_team_venue_gap': home_team_venue_gap,
                'away_team_venue_gap': away_team_venue_gap,
            }
            features_list.append(features)

            home_goals = match['ft_home']
            away_goals = match['ft_away']

            if home_goals > away_goals:
                home_actual, away_actual = 1.0, 0.0
            elif home_goals < away_goals:
                home_actual, away_actual = 0.0, 1.0
            else:
                home_actual, away_actual = 0.5, 0.5

            # Update overall ratings
            team_ratings[home_id] = home_elo + self.k_factor * (home_actual - home_expected)
            team_ratings[away_id] = away_elo + self.k_factor * (away_actual - away_expected)

            # Update venue-specific ratings
            venue_home_expected = self._expected_score(
                home_venue_elo + self.home_advantage, away_venue_elo
            )
            home_venue_ratings[home_id] = home_venue_elo + self.k_factor * (
                home_actual - venue_home_expected
            )
            away_venue_ratings[away_id] = away_venue_elo + self.k_factor * (
                away_actual - (1 - venue_home_expected)
            )

        print(f"Created {len(features_list)} ELO rating features (K={self.k_factor}, with venue-specific)")
        return pd.DataFrame(features_list)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A against team B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))



class PoissonFeatureEngineer(BaseFeatureEngineer):
    """
    Creates Poisson-based features for goal prediction.

    Uses historical goal-scoring rates to estimate:
    - Expected goals for each team
    - Match outcome probabilities based on Poisson distribution

    The Poisson model assumes:
    - Goals are independent events
    - Average rate (lambda) based on attack/defense strength
    """

    def __init__(self, lookback_matches: int = 10):
        """
        Args:
            lookback_matches: Number of recent matches to calculate rates
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate Poisson-based features.

        Args:
            data: dict with 'matches' DataFrame

        Returns:
            DataFrame with Poisson features
        """
        from scipy.stats import poisson

        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Calculate league averages (will be updated as we go)
        all_teams = set(matches['home_team_id'].unique()) | set(matches['away_team_id'].unique())

        # Track team stats
        team_stats = {
            team_id: {
                'goals_scored': [],
                'goals_conceded': [],
                'home_goals_scored': [],
                'home_goals_conceded': [],
                'away_goals_scored': [],
                'away_goals_conceded': [],
            }
            for team_id in all_teams
        }

        features_list = []

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            home_attack = self._get_attack_strength(team_stats, home_id, is_home=True)
            home_defense = self._get_defense_strength(team_stats, home_id, is_home=True)
            away_attack = self._get_attack_strength(team_stats, away_id, is_home=False)
            away_defense = self._get_defense_strength(team_stats, away_id, is_home=False)

            # NOTE: Baselines calibrated to actual league averages
            # Feature-level xG underestimates slightly (-0.15), but this is acceptable
            # Overconfidence in predictions comes from ML model outputs, addressed via
            # calibration module in src/calibration/market_calibrator.py
            home_xg = max(0.5, home_attack - away_defense + 1.5)  # baseline ~1.5 goals
            away_xg = max(0.5, away_attack - home_defense + 1.0)   # away teams score less

            home_win_prob, draw_prob, away_win_prob = self._poisson_outcome_probs(home_xg, away_xg)

            features = {
                'fixture_id': match['fixture_id'],
                'home_xg_poisson': home_xg,
                'away_xg_poisson': away_xg,
                'xg_diff': home_xg - away_xg,
                'home_attack_strength': home_attack,
                'home_defense_strength': home_defense,
                'away_attack_strength': away_attack,
                'away_defense_strength': away_defense,
                'poisson_home_win_prob': home_win_prob,
                'poisson_draw_prob': draw_prob,
                'poisson_away_win_prob': away_win_prob,
            }
            features_list.append(features)

            home_goals = match['ft_home']
            away_goals = match['ft_away']

            team_stats[home_id]['goals_scored'].append(home_goals)
            team_stats[home_id]['goals_conceded'].append(away_goals)
            team_stats[home_id]['home_goals_scored'].append(home_goals)
            team_stats[home_id]['home_goals_conceded'].append(away_goals)

            team_stats[away_id]['goals_scored'].append(away_goals)
            team_stats[away_id]['goals_conceded'].append(home_goals)
            team_stats[away_id]['away_goals_scored'].append(away_goals)
            team_stats[away_id]['away_goals_conceded'].append(home_goals)

        print(f"Created {len(features_list)} Poisson features (lookback={self.lookback_matches})")
        return pd.DataFrame(features_list)

    def _get_attack_strength(self, team_stats: Dict, team_id: int, is_home: bool) -> float:
        """Calculate attack strength (goals scored per match - league avg)."""
        stats = team_stats[team_id]
        key = 'home_goals_scored' if is_home else 'away_goals_scored'
        goals = stats[key][-self.lookback_matches:] if stats[key] else []

        if not goals:
            return 0.0

        return np.mean(goals) - 1.3

    def _get_defense_strength(self, team_stats: Dict, team_id: int, is_home: bool) -> float:
        """Calculate defense strength (goals conceded per match - league avg)."""
        stats = team_stats[team_id]
        key = 'home_goals_conceded' if is_home else 'away_goals_conceded'
        goals = stats[key][-self.lookback_matches:] if stats[key] else []

        if not goals:
            return 0.0

        return np.mean(goals) - 1.3

    def _poisson_outcome_probs(self, home_xg: float, away_xg: float, max_goals: int = 7) -> tuple:
        """
        Calculate match outcome probabilities using Poisson distribution.

        Returns:
            (home_win_prob, draw_prob, away_win_prob)
        """
        from scipy.stats import poisson

        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0

        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)

                if home_goals > away_goals:
                    home_win_prob += prob
                elif home_goals == away_goals:
                    draw_prob += prob
                else:
                    away_win_prob += prob

        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total

        return home_win_prob, draw_prob, away_win_prob



class TeamRatingFeatureEngineer(BaseFeatureEngineer):
    """
    Creates features based on average team ratings.

    Uses player ratings to calculate overall team strength.
    """

    def __init__(self, lookback_matches: int = 5):
        """
        Args:
            lookback_matches: Number of recent matches for rating calculation
        """
        self.lookback_matches = lookback_matches

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate team rating features.
        """
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)
        player_stats = data.get('player_stats')

        if player_stats is None or player_stats.empty:
            print("No player stats data available, skipping team rating features")
            return pd.DataFrame({'fixture_id': matches['fixture_id']})

        # Track team ratings history
        team_ratings_history = {}  # {team_id: [avg_ratings]}
        features_list = []

        for idx, match in matches.iterrows():
            fixture_id = match['fixture_id']
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Get historical team ratings
            home_ratings = team_ratings_history.get(home_id, [])
            away_ratings = team_ratings_history.get(away_id, [])

            home_avg = sum(home_ratings) / len(home_ratings) if home_ratings else 6.5
            away_avg = sum(away_ratings) / len(away_ratings) if away_ratings else 6.5

            features = {
                'fixture_id': fixture_id,
                'home_team_avg_rating': home_avg,
                'away_team_avg_rating': away_avg,
                'team_rating_diff': home_avg - away_avg,
            }
            features_list.append(features)

            # Update team ratings from this match
            match_stats = player_stats[
                (player_stats['fixture_id'] == fixture_id) &
                (player_stats['rating'].notna()) &
                (player_stats['rating'] > 0)
            ]

            for team_id in [home_id, away_id]:
                team_match_ratings = match_stats[
                    match_stats['team_id'] == team_id
                ]['rating'].tolist()

                if team_match_ratings:
                    avg_rating = sum(team_match_ratings) / len(team_match_ratings)

                    if team_id not in team_ratings_history:
                        team_ratings_history[team_id] = []
                    team_ratings_history[team_id].append(avg_rating)

                    if len(team_ratings_history[team_id]) > self.lookback_matches:
                        team_ratings_history[team_id].pop(0)

        print(f"Created {len(features_list)} team rating features")
        return pd.DataFrame(features_list)


class PoissonGLMFeatureEngineer(BaseFeatureEngineer):
    """
    Creates Poisson GLM-based features for goal prediction.

    Unlike the simple PoissonFeatureEngineer which uses fixed baselines
    (1.3 league avg, 1.5/1.0 home boost), this engineer fits two Poisson
    GLMs on historical matches to learn team-specific attack/defense
    coefficients and a data-driven home advantage.

    Model specification:
        home_goals ~ C(home_team) + C(away_team) + home_advantage
        away_goals ~ C(home_team) + C(away_team)

    Teams with fewer than min_matches_per_team get the league average
    coefficient (natural shrinkage toward mean).

    The existing simple PoissonFeatureEngineer is kept as-is — this adds
    complementary features (glm_ prefix) so the ML models can learn which
    Poisson approach is more useful.
    """

    def __init__(
        self,
        lookback_days: int = 365,
        min_matches_per_team: int = 10,
        refit_every: int = 50,
    ):
        """
        Args:
            lookback_days: Only use matches within this many days for fitting.
            min_matches_per_team: Minimum matches before a team gets its own
                coefficient. Below this, the team uses the league intercept.
            refit_every: Refit the GLM every N matches (for efficiency).
        """
        self.lookback_days = lookback_days
        self.min_matches_per_team = min_matches_per_team
        self.refit_every = refit_every

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate Poisson GLM-based features.

        Args:
            data: dict with 'matches' DataFrame.

        Returns:
            DataFrame with GLM xG features.
        """
        from scipy.stats import poisson

        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(matches['date']):
            matches['date'] = pd.to_datetime(matches['date'])

        features_list = []

        # GLM state: fitted coefficients
        team_attack: Dict[int, float] = {}   # team_id -> attack strength
        team_defense: Dict[int, float] = {}  # team_id -> defense strength
        home_adv: float = 0.0
        league_avg: float = 1.3
        last_fit_idx: int = -self.refit_every  # Force fit on first batch

        for idx, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            match_date = match['date']

            # Refit GLM periodically
            if idx - last_fit_idx >= self.refit_every:
                cutoff = match_date - pd.Timedelta(days=self.lookback_days)
                train_df = matches.loc[:idx - 1]  # All matches before current
                train_df = train_df[train_df['date'] >= cutoff]

                if len(train_df) >= 30:
                    result = self._fit_glm(train_df)
                    if result is not None:
                        team_attack, team_defense, home_adv, league_avg = result
                        last_fit_idx = idx

            # Compute xG from GLM coefficients
            h_att = team_attack.get(home_id, 0.0)
            h_def = team_defense.get(home_id, 0.0)
            a_att = team_attack.get(away_id, 0.0)
            a_def = team_defense.get(away_id, 0.0)

            # Expected goals: exp(intercept + attack_home + defense_away + home_adv)
            glm_home_xg = max(0.3, np.exp(np.log(max(league_avg, 0.5)) + h_att - a_def + home_adv))
            glm_away_xg = max(0.3, np.exp(np.log(max(league_avg, 0.5)) + a_att - h_def))

            # Compute outcome probabilities from Poisson
            h_win, draw, a_win = self._poisson_outcome_probs(glm_home_xg, glm_away_xg)

            features = {
                'fixture_id': match['fixture_id'],
                'glm_home_xg': round(glm_home_xg, 4),
                'glm_away_xg': round(glm_away_xg, 4),
                'glm_xg_diff': round(glm_home_xg - glm_away_xg, 4),
                'glm_home_attack': round(h_att, 4),
                'glm_away_attack': round(a_att, 4),
                'glm_home_defense': round(h_def, 4),
                'glm_away_defense': round(a_def, 4),
                'glm_home_advantage': round(home_adv, 4),
                'glm_home_win_prob': round(h_win, 4),
                'glm_draw_prob': round(draw, 4),
                'glm_away_win_prob': round(a_win, 4),
            }
            features_list.append(features)

        print(f"Created {len(features_list)} Poisson GLM features "
              f"(lookback={self.lookback_days}d, refit_every={self.refit_every})")
        return pd.DataFrame(features_list)

    def _fit_glm(
        self, train_df: pd.DataFrame
    ) -> Optional[Tuple[Dict[int, float], Dict[int, float], float, float]]:
        """
        Fit Poisson GLM on training data and extract team coefficients.

        Returns (team_attack, team_defense, home_advantage, league_avg) or None on failure.
        """
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.families import Poisson
        except ImportError:
            logger.warning("statsmodels not installed — falling back to simple Poisson")
            return None

        # Count matches per team
        home_counts = train_df['home_team_id'].value_counts()
        away_counts = train_df['away_team_id'].value_counts()
        total_counts = home_counts.add(away_counts, fill_value=0)

        # Only include teams with enough matches
        eligible_teams = set(total_counts[total_counts >= self.min_matches_per_team].index)

        if len(eligible_teams) < 4:
            return None

        # Filter to matches where both teams are eligible
        mask = (
            train_df['home_team_id'].isin(eligible_teams)
            & train_df['away_team_id'].isin(eligible_teams)
        )
        fit_df = train_df[mask].copy()

        if len(fit_df) < 20:
            return None

        # Build long-format data for GLM: each match contributes 2 rows
        # (home goals, away goals)
        team_ids = sorted(eligible_teams)
        team_to_idx = {t: i for i, t in enumerate(team_ids)}
        n_teams = len(team_ids)

        rows_goals = []
        rows_attack = []  # one-hot for attacking team
        rows_defense = []  # one-hot for defending team
        rows_home = []

        for _, match in fit_df.iterrows():
            h_id = match['home_team_id']
            a_id = match['away_team_id']
            h_goals = match.get('ft_home', 0)
            a_goals = match.get('ft_away', 0)

            if pd.isna(h_goals) or pd.isna(a_goals):
                continue

            h_idx = team_to_idx[h_id]
            a_idx = team_to_idx[a_id]

            # Home team scoring row
            att_vec = np.zeros(n_teams)
            def_vec = np.zeros(n_teams)
            att_vec[h_idx] = 1
            def_vec[a_idx] = 1
            rows_goals.append(int(h_goals))
            rows_attack.append(att_vec)
            rows_defense.append(def_vec)
            rows_home.append(1)

            # Away team scoring row
            att_vec2 = np.zeros(n_teams)
            def_vec2 = np.zeros(n_teams)
            att_vec2[a_idx] = 1
            def_vec2[h_idx] = 1
            rows_goals.append(int(a_goals))
            rows_attack.append(att_vec2)
            rows_defense.append(def_vec2)
            rows_home.append(0)

        if len(rows_goals) < 40:
            return None

        y = np.array(rows_goals)
        X_attack = np.array(rows_attack)[:, 1:]  # Drop first team (reference)
        X_defense = np.array(rows_defense)[:, 1:]
        X_home = np.array(rows_home).reshape(-1, 1)
        X = np.hstack([X_attack, X_defense, X_home])
        X = sm.add_constant(X)

        try:
            model = sm.GLM(y, X, family=Poisson())
            result = model.fit(disp=0, maxiter=50)
        except Exception as e:
            logger.debug(f"GLM fit failed: {e}")
            return None

        params = result.params
        intercept = params[0]
        league_avg = np.exp(intercept)

        # Extract coefficients
        attack_coefs = np.concatenate([[0.0], params[1:n_teams]])  # ref team = 0
        defense_coefs = np.concatenate([[0.0], params[n_teams:2 * n_teams - 1]])
        home_advantage = params[-1]

        team_attack = {team_ids[i]: float(attack_coefs[i]) for i in range(n_teams)}
        team_defense = {team_ids[i]: float(defense_coefs[i]) for i in range(n_teams)}

        return team_attack, team_defense, float(home_advantage), float(league_avg)

    @staticmethod
    def _poisson_outcome_probs(
        home_xg: float, away_xg: float, max_goals: int = 7
    ) -> Tuple[float, float, float]:
        """Calculate match outcome probabilities from Poisson xG."""
        from scipy.stats import poisson

        h_win = d = a_win = 0.0
        for hg in range(max_goals):
            for ag in range(max_goals):
                p = poisson.pmf(hg, home_xg) * poisson.pmf(ag, away_xg)
                if hg > ag:
                    h_win += p
                elif hg == ag:
                    d += p
                else:
                    a_win += p

        total = h_win + d + a_win
        if total > 0:
            h_win /= total
            d /= total
            a_win /= total
        return h_win, d, a_win

    def get_feature_names(self) -> List[str]:
        """Return list of feature names created by this engineer."""
        return [
            'glm_home_xg', 'glm_away_xg', 'glm_xg_diff',
            'glm_home_attack', 'glm_away_attack',
            'glm_home_defense', 'glm_away_defense',
            'glm_home_advantage',
            'glm_home_win_prob', 'glm_draw_prob', 'glm_away_win_prob',
        ]
