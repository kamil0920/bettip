"""
Bivariate Poisson (Dixon-Coles) model for correct score predictions.

The Dixon-Coles model extends the standard bivariate Poisson to add
a dependence parameter (tau) that corrects for the correlation between
low-scoring games. This is important because 0-0 and 1-1 draws are more
common than a simple Poisson model would predict.

Reference:
Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores
and inefficiencies in the football betting market.

Features:
- Predicts expected goals (xG) for home and away teams
- Calculates full scoreline probability matrix (0-0 to 10-10)
- Applies Dixon-Coles low-score adjustment
- Supports Over/Under, BTTS, Correct Score, and Match Result markets
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class MatchPrediction:
    """Container for match prediction results."""
    home_xg: float
    away_xg: float
    score_matrix: np.ndarray  # Probability matrix for each scoreline
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_1_5_prob: float
    over_2_5_prob: float
    over_3_5_prob: float
    btts_prob: float
    correct_score_probs: Dict[str, float]  # Top scoreline probabilities


class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model for football score prediction.

    The model estimates:
    - Attack and defense ratings for each team
    - Home advantage factor
    - Rho (ρ) parameter for low-score dependence correction

    The expected goals are:
        home_xG = exp(attack_home - defense_away + home_advantage)
        away_xG = exp(attack_away - defense_home)
    """

    def __init__(self, max_goals: int = 10, time_decay: float = 0.0018):
        """
        Args:
            max_goals: Maximum goals to consider in probability matrix
            time_decay: Exponential decay factor for weighting recent matches
                       Default 0.0018 gives half-weight to matches 385 days ago
        """
        self.max_goals = max_goals
        self.time_decay = time_decay
        self.teams_ = None
        self.attack_ = None
        self.defense_ = None
        self.home_adv_ = None
        self.rho_ = None

    def _tau(self, home_goals: int, away_goals: int, home_xg: float, away_xg: float, rho: float) -> float:
        """
        Dixon-Coles adjustment factor for low-scoring games.

        This corrects the independence assumption of Poisson for scores 0-0, 1-0, 0-1, 1-1.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0

    def _score_probability(self, home_goals: int, away_goals: int,
                          home_xg: float, away_xg: float, rho: float) -> float:
        """Calculate probability of a specific scoreline."""
        base_prob = (poisson.pmf(home_goals, home_xg) *
                    poisson.pmf(away_goals, away_xg))
        tau = self._tau(home_goals, away_goals, home_xg, away_xg, rho)
        return base_prob * tau

    def _neg_log_likelihood(self, params: np.ndarray, matches: pd.DataFrame,
                           team_to_idx: Dict[str, int], weights: np.ndarray) -> float:
        """Negative log-likelihood for parameter optimization."""
        n_teams = len(team_to_idx)
        attack = params[:n_teams]
        defense = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1]

        log_likelihood = 0.0

        for idx, match in matches.iterrows():
            home_idx = team_to_idx[match['home_team']]
            away_idx = team_to_idx[match['away_team']]

            home_xg = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
            away_xg = np.exp(attack[away_idx] - defense[home_idx])

            # Clip to avoid numerical issues
            home_xg = np.clip(home_xg, 0.01, 10)
            away_xg = np.clip(away_xg, 0.01, 10)

            home_goals = int(match['home_goals'])
            away_goals = int(match['away_goals'])

            prob = self._score_probability(home_goals, away_goals, home_xg, away_xg, rho)
            prob = max(prob, 1e-10)

            weight = weights[idx] if idx < len(weights) else 1.0
            log_likelihood += weight * np.log(prob)

        return -log_likelihood

    def fit(self, matches: pd.DataFrame,
           home_col: str = 'home_team',
           away_col: str = 'away_team',
           home_goals_col: str = 'home_goals',
           away_goals_col: str = 'away_goals',
           date_col: str = 'date') -> 'DixonColesModel':
        """
        Fit the Dixon-Coles model to historical match data.

        Args:
            matches: DataFrame with match results
            home_col: Column name for home team
            away_col: Column name for away team
            home_goals_col: Column name for home goals
            away_goals_col: Column name for away goals
            date_col: Column name for match date (for time decay)

        Returns:
            self
        """
        # Prepare data
        df = matches.copy()
        df = df.rename(columns={
            home_col: 'home_team',
            away_col: 'away_team',
            home_goals_col: 'home_goals',
            away_goals_col: 'away_goals',
            date_col: 'date'
        })

        # Get unique teams
        teams = list(set(df['home_team'].tolist() + df['away_team'].tolist()))
        team_to_idx = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        # Calculate time weights (more recent matches get higher weight)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            max_date = df['date'].max()
            days_ago = (max_date - df['date']).dt.days
            weights = np.exp(-self.time_decay * days_ago)
        else:
            weights = np.ones(len(df))

        # Initialize parameters
        # attack, defense for each team, home_advantage, rho
        n_params = 2 * n_teams + 2
        x0 = np.zeros(n_params)
        x0[2*n_teams] = 0.25  # Initial home advantage
        x0[2*n_teams + 1] = 0.0  # Initial rho

        # Bounds
        bounds = [(None, None)] * (2 * n_teams)  # attack and defense
        bounds.append((0.0, 1.0))  # home advantage
        bounds.append((-1.0, 1.0))  # rho

        # Constraint: sum of attack ratings = 0 (for identifiability)
        def attack_sum_constraint(params):
            return np.sum(params[:n_teams])

        constraints = [{'type': 'eq', 'fun': attack_sum_constraint}]

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._neg_log_likelihood,
                x0,
                args=(df, team_to_idx, weights),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'disp': False}
            )

        # Store results
        self.teams_ = teams
        self.team_to_idx_ = team_to_idx
        self.attack_ = dict(zip(teams, result.x[:n_teams]))
        self.defense_ = dict(zip(teams, result.x[n_teams:2*n_teams]))
        self.home_adv_ = result.x[2*n_teams]
        self.rho_ = result.x[2*n_teams + 1]

        return self

    def predict_xg(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Predict expected goals for a match.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Tuple of (home_xg, away_xg)
        """
        if self.attack_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use average ratings for unknown teams
        avg_attack = np.mean(list(self.attack_.values()))
        avg_defense = np.mean(list(self.defense_.values()))

        home_attack = self.attack_.get(home_team, avg_attack)
        home_defense = self.defense_.get(home_team, avg_defense)
        away_attack = self.attack_.get(away_team, avg_attack)
        away_defense = self.defense_.get(away_team, avg_defense)

        home_xg = np.exp(home_attack - away_defense + self.home_adv_)
        away_xg = np.exp(away_attack - home_defense)

        return home_xg, away_xg

    def predict_score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """
        Calculate full scoreline probability matrix.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            2D array where [i,j] is P(home_goals=i, away_goals=j)
        """
        home_xg, away_xg = self.predict_xg(home_team, away_team)

        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))

        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                matrix[i, j] = self._score_probability(i, j, home_xg, away_xg, self.rho_)

        # Normalize to sum to 1
        matrix /= matrix.sum()

        return matrix

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        """
        Generate full match prediction with all betting markets.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            MatchPrediction object with all probabilities
        """
        home_xg, away_xg = self.predict_xg(home_team, away_team)
        score_matrix = self.predict_score_matrix(home_team, away_team)

        # Match result probabilities
        home_win_prob = np.sum(np.tril(score_matrix, -1))  # Below diagonal
        draw_prob = np.sum(np.diag(score_matrix))  # Diagonal
        away_win_prob = np.sum(np.triu(score_matrix, 1))  # Above diagonal

        # Total goals probabilities
        total_goals_prob = np.zeros(2 * self.max_goals + 1)
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                total_goals_prob[i + j] += score_matrix[i, j]

        over_1_5 = 1 - sum(total_goals_prob[:2])
        over_2_5 = 1 - sum(total_goals_prob[:3])
        over_3_5 = 1 - sum(total_goals_prob[:4])

        # BTTS probability (both teams score)
        btts_prob = 1 - (score_matrix[0, :].sum() + score_matrix[:, 0].sum() - score_matrix[0, 0])

        # Top correct scores
        correct_scores = {}
        for i in range(min(6, self.max_goals + 1)):
            for j in range(min(6, self.max_goals + 1)):
                score_str = f"{i}-{j}"
                correct_scores[score_str] = float(score_matrix[i, j])

        # Sort by probability
        correct_scores = dict(sorted(correct_scores.items(),
                                     key=lambda x: x[1], reverse=True)[:10])

        return MatchPrediction(
            home_xg=home_xg,
            away_xg=away_xg,
            score_matrix=score_matrix,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            over_1_5_prob=over_1_5,
            over_2_5_prob=over_2_5,
            over_3_5_prob=over_3_5,
            btts_prob=btts_prob,
            correct_score_probs=correct_scores
        )

    def get_team_ratings(self) -> pd.DataFrame:
        """Get attack and defense ratings for all teams."""
        if self.attack_ is None:
            raise ValueError("Model not fitted.")

        df = pd.DataFrame({
            'team': self.teams_,
            'attack': [self.attack_[t] for t in self.teams_],
            'defense': [self.defense_[t] for t in self.teams_],
        })
        df['attack_rank'] = df['attack'].rank(ascending=False)
        df['defense_rank'] = df['defense'].rank(ascending=True)  # Lower is better
        df['overall'] = df['attack'] - df['defense']
        df = df.sort_values('overall', ascending=False)

        return df


def calculate_correct_score_ev(prediction: MatchPrediction, odds: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate expected value for correct score bets.

    Args:
        prediction: MatchPrediction object
        odds: Dict mapping scorelines to decimal odds (e.g., {'1-0': 6.5, '2-1': 8.0})

    Returns:
        Dict mapping scorelines to expected value
    """
    ev_dict = {}

    for score, prob in prediction.correct_score_probs.items():
        if score in odds:
            odds_val = odds[score]
            ev = prob * odds_val - 1  # Expected value per unit staked
            ev_dict[score] = ev

    return ev_dict


def find_value_correct_scores(prediction: MatchPrediction, odds: Dict[str, float],
                              min_ev: float = 0.05) -> List[Dict]:
    """
    Find correct score bets with positive expected value.

    Args:
        prediction: MatchPrediction object
        odds: Dict mapping scorelines to decimal odds
        min_ev: Minimum expected value threshold

    Returns:
        List of value bets with scoreline, probability, odds, and EV
    """
    value_bets = []

    for score, prob in prediction.correct_score_probs.items():
        if score in odds:
            odds_val = odds[score]
            implied_prob = 1 / odds_val
            ev = prob * odds_val - 1

            if ev >= min_ev and prob > implied_prob:
                value_bets.append({
                    'scoreline': score,
                    'model_prob': prob,
                    'implied_prob': implied_prob,
                    'odds': odds_val,
                    'edge': prob - implied_prob,
                    'ev': ev
                })

    return sorted(value_bets, key=lambda x: x['ev'], reverse=True)


if __name__ == "__main__":
    # Example usage
    print("Dixon-Coles Model Example")
    print("=" * 50)

    # Create sample match data
    np.random.seed(42)
    teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F']
    n_matches = 100

    matches = []
    for _ in range(n_matches):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        # Simulate goals with Poisson
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        matches.append({
            'home_team': home,
            'away_team': away,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        })

    df = pd.DataFrame(matches)

    # Fit model
    model = DixonColesModel()
    model.fit(df)

    print(f"Home advantage: {model.home_adv_:.4f}")
    print(f"Rho (low-score adjustment): {model.rho_:.4f}")
    print()

    # Get team ratings
    ratings = model.get_team_ratings()
    print("Team Ratings:")
    print(ratings.to_string(index=False))
    print()

    # Predict a match
    prediction = model.predict('Team A', 'Team B')
    print(f"\nPrediction: Team A vs Team B")
    print(f"Expected goals: {prediction.home_xg:.2f} - {prediction.away_xg:.2f}")
    print(f"Match result: H={prediction.home_win_prob:.1%}, D={prediction.draw_prob:.1%}, A={prediction.away_win_prob:.1%}")
    print(f"Over 2.5: {prediction.over_2_5_prob:.1%}")
    print(f"BTTS: {prediction.btts_prob:.1%}")
    print(f"Top scorelines: {prediction.correct_score_probs}")
