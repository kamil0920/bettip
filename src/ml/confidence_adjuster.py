"""
Confidence adjustment layer for pre-match lineup information.

Adjusts ML model predictions based on confirmed lineup data.
When lineups are announced (~1hr before match), this layer:
1. Compares expected vs actual key player availability
2. Boosts confidence when lineups confirm model assumptions
3. Reduces confidence when lineups contradict assumptions

This is the final adjustment layer before bet placement.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Key players by team - high impact on match outcomes
# Based on goals/assists contribution, market value, and tactical importance
KEY_PLAYERS = {
    # Premier League
    33: {  # Man Utd
        'core': ['Bruno Fernandes', 'Kobbie Mainoo'],
        'scoring': ['Marcus Rashford', 'Rasmus Hojlund'],
        'defensive': ['Lisandro Martinez', 'Andre Onana'],
    },
    40: {  # Liverpool
        'core': ['Virgil van Dijk', 'Trent Alexander-Arnold'],
        'scoring': ['Mohamed Salah', 'Luis Diaz'],
        'defensive': ['Alisson'],
    },
    50: {  # Man City
        'core': ['Rodri', 'Kevin De Bruyne'],
        'scoring': ['Erling Haaland', 'Phil Foden'],
        'defensive': ['Ederson', 'Ruben Dias'],
    },
    42: {  # Arsenal
        'core': ['Martin Odegaard', 'Declan Rice'],
        'scoring': ['Bukayo Saka', 'Kai Havertz'],
        'defensive': ['William Saliba', 'Gabriel'],
    },
    49: {  # Chelsea
        'core': ['Enzo Fernandez', 'Moises Caicedo'],
        'scoring': ['Cole Palmer', 'Nicolas Jackson'],
        'defensive': ['Reece James'],
    },
    47: {  # Tottenham
        'core': ['James Maddison'],
        'scoring': ['Son Heung-min', 'Dominic Solanke'],
        'defensive': ['Cristian Romero'],
    },
    34: {  # Newcastle
        'core': ['Bruno Guimaraes', 'Joelinton'],
        'scoring': ['Alexander Isak', 'Anthony Gordon'],
        'defensive': ['Nick Pope'],
    },
    66: {  # Aston Villa
        'core': ['John McGinn', 'Youri Tielemans'],
        'scoring': ['Ollie Watkins'],
        'defensive': ['Emiliano Martinez'],
    },

    # La Liga
    529: {  # Real Madrid
        'core': ['Jude Bellingham', 'Aurelien Tchouameni'],
        'scoring': ['Vinicius Junior', 'Kylian Mbappe', 'Rodrygo'],
        'defensive': ['Thibaut Courtois', 'Antonio Rudiger'],
    },
    530: {  # Barcelona
        'core': ['Pedri', 'Gavi', 'Frenkie de Jong'],
        'scoring': ['Lamine Yamal', 'Robert Lewandowski', 'Raphinha'],
        'defensive': ['Marc-Andre ter Stegen'],
    },
    531: {  # Atletico Madrid
        'core': ['Koke'],
        'scoring': ['Antoine Griezmann', 'Alvaro Morata'],
        'defensive': ['Jan Oblak'],
    },

    # Serie A
    489: {  # Inter
        'core': ['Nicolo Barella', 'Hakan Calhanoglu'],
        'scoring': ['Lautaro Martinez', 'Marcus Thuram'],
        'defensive': ['Alessandro Bastoni'],
    },
    492: {  # AC Milan
        'core': ['Tijjani Reijnders'],
        'scoring': ['Rafael Leao', 'Christian Pulisic'],
        'defensive': ['Theo Hernandez', 'Mike Maignan'],
    },
    496: {  # Juventus
        'core': ['Manuel Locatelli'],
        'scoring': ['Dusan Vlahovic', 'Kenan Yildiz'],
        'defensive': ['Gleison Bremer'],
    },

    # Bundesliga
    157: {  # Bayern Munich
        'core': ['Joshua Kimmich', 'Jamal Musiala'],
        'scoring': ['Harry Kane', 'Leroy Sane', 'Serge Gnabry'],
        'defensive': ['Dayot Upamecano', 'Manuel Neuer'],
    },
    165: {  # Leverkusen
        'core': ['Florian Wirtz', 'Granit Xhaka'],
        'scoring': ['Victor Boniface'],
        'defensive': [],
    },

    # Ligue 1
    85: {  # PSG
        'core': ['Vitinha', 'Warren Zaire-Emery'],
        'scoring': ['Ousmane Dembele', 'Bradley Barcola', 'Goncalo Ramos'],
        'defensive': ['Achraf Hakimi', 'Gianluigi Donnarumma'],
    },
}

# Impact weights by player category
CATEGORY_WEIGHTS = {
    'core': 1.0,      # Highest impact - tactical lynchpin
    'scoring': 0.85,  # Direct goal threat
    'defensive': 0.7, # Defensive stability
}

# Position impact on different markets
MARKET_POSITION_IMPACT = {
    'home_win': {'scoring': 0.8, 'core': 0.7, 'defensive': 0.5},
    'away_win': {'defensive': 0.6, 'core': 0.7, 'scoring': 0.7},
    'btts': {'scoring': 0.9, 'defensive': 0.8, 'core': 0.5},
    'over25': {'scoring': 0.9, 'core': 0.6, 'defensive': 0.4},
    'under25': {'defensive': 0.9, 'scoring': 0.4, 'core': 0.5},
}


@dataclass
class LineupAnalysis:
    """Result of lineup analysis for a team."""
    team_id: int
    team_name: str
    key_players_available: List[str]
    key_players_missing: List[str]
    formation: Optional[str]
    strength_score: float  # 0-1, 1 being full strength
    missing_impact: Dict[str, float]  # Impact by category


@dataclass
class ConfidenceAdjustment:
    """Confidence adjustment recommendation."""
    fixture_id: int
    market: str
    original_probability: float
    adjusted_probability: float
    adjustment_factor: float  # >1 = boost, <1 = reduce
    confidence_change: str  # 'boost', 'reduce', 'neutral'
    reasons: List[str]


class LineupConfidenceAdjuster:
    """
    Adjusts prediction confidence based on confirmed lineups.

    This is the final adjustment layer before bet placement.
    It compares expected team strength (from training features) with
    actual lineup information.
    """

    def __init__(
        self,
        key_players: Optional[Dict[int, Dict[str, List[str]]]] = None,
        max_adjustment: float = 0.15,  # Max probability adjustment
        key_player_threshold: int = 2,  # Min missing key players for adjustment
    ):
        """
        Initialize adjuster.

        Args:
            key_players: Dict mapping team_id to categorized player lists
            max_adjustment: Maximum probability adjustment (default 0.15 = 15%)
            key_player_threshold: Number of key players missing to trigger adjustment
        """
        self.key_players = key_players or KEY_PLAYERS
        self.max_adjustment = max_adjustment
        self.key_player_threshold = key_player_threshold

    def analyze_lineup(
        self,
        team_id: int,
        team_name: str,
        lineup: Dict[str, Any],
    ) -> LineupAnalysis:
        """
        Analyze a team's lineup against expected key players.

        Args:
            team_id: Team ID
            team_name: Team name
            lineup: Lineup data from API (contains 'startXI' with player list)

        Returns:
            LineupAnalysis with availability and impact scores
        """
        # Get key players for this team
        team_key_players = self.key_players.get(team_id, {})
        if not team_key_players:
            return LineupAnalysis(
                team_id=team_id,
                team_name=team_name,
                key_players_available=[],
                key_players_missing=[],
                formation=lineup.get('formation'),
                strength_score=1.0,
                missing_impact={},
            )

        # Extract starting XI names
        starting_xi = []
        if 'startXI' in lineup:
            for player_entry in lineup['startXI']:
                player = player_entry.get('player', {})
                name = player.get('name', '')
                if name:
                    starting_xi.append(name)

        # Check each key player category
        available = []
        missing = []
        missing_impact = {}

        for category, players in team_key_players.items():
            category_missing = []
            for player_name in players:
                # Check if player is in starting XI (fuzzy match)
                found = any(
                    player_name.lower() in xi_player.lower() or
                    xi_player.lower() in player_name.lower()
                    for xi_player in starting_xi
                )
                if found:
                    available.append(player_name)
                else:
                    missing.append(player_name)
                    category_missing.append(player_name)

            if category_missing:
                weight = CATEGORY_WEIGHTS.get(category, 0.5)
                impact = len(category_missing) / len(players) * weight
                missing_impact[category] = impact

        # Calculate strength score (1.0 = full strength)
        total_key = sum(len(p) for p in team_key_players.values())
        if total_key > 0:
            strength_score = len(available) / total_key
        else:
            strength_score = 1.0

        return LineupAnalysis(
            team_id=team_id,
            team_name=team_name,
            key_players_available=available,
            key_players_missing=missing,
            formation=lineup.get('formation'),
            strength_score=strength_score,
            missing_impact=missing_impact,
        )

    def calculate_adjustment(
        self,
        fixture_id: int,
        market: str,
        original_prob: float,
        home_analysis: LineupAnalysis,
        away_analysis: LineupAnalysis,
    ) -> ConfidenceAdjustment:
        """
        Calculate probability adjustment based on lineup analysis.

        Args:
            fixture_id: Fixture ID
            market: Betting market ('home_win', 'away_win', 'btts', 'over25')
            original_prob: Original model probability
            home_analysis: Home team lineup analysis
            away_analysis: Away team lineup analysis

        Returns:
            ConfidenceAdjustment with adjusted probability
        """
        reasons = []
        adjustment_factor = 1.0

        # Get market-specific position impacts
        market_impacts = MARKET_POSITION_IMPACT.get(market, {
            'scoring': 0.7, 'core': 0.7, 'defensive': 0.7
        })

        # Calculate impact based on market
        if market == 'home_win':
            # Home strength matters more, away defense less
            home_impact = self._calculate_team_impact(
                home_analysis, market_impacts, positive=True)
            away_impact = self._calculate_team_impact(
                away_analysis, market_impacts, positive=False)
            total_impact = home_impact + away_impact

            if len(home_analysis.key_players_missing) >= self.key_player_threshold:
                reasons.append(
                    f"Home team missing {len(home_analysis.key_players_missing)} "
                    f"key players: {', '.join(home_analysis.key_players_missing[:3])}"
                )
            if len(away_analysis.key_players_missing) >= self.key_player_threshold:
                reasons.append(
                    f"Away team weakened ({len(away_analysis.key_players_missing)} "
                    f"key players missing)"
                )

        elif market == 'away_win':
            # Away strength matters more
            away_impact = self._calculate_team_impact(
                away_analysis, market_impacts, positive=True)
            home_impact = self._calculate_team_impact(
                home_analysis, market_impacts, positive=False)
            total_impact = away_impact + home_impact

            if len(away_analysis.key_players_missing) >= self.key_player_threshold:
                reasons.append(
                    f"Away team missing {len(away_analysis.key_players_missing)} "
                    f"key players"
                )
            if len(home_analysis.key_players_missing) >= self.key_player_threshold:
                reasons.append(
                    f"Home team weakened, favors away"
                )

        elif market == 'btts':
            # Both teams' scoring ability matters
            home_scoring = home_analysis.missing_impact.get('scoring', 0)
            away_scoring = away_analysis.missing_impact.get('scoring', 0)
            total_impact = -(home_scoring + away_scoring) * 0.5  # Negative = reduce

            if home_scoring > 0:
                reasons.append(f"Home team missing key scorers")
            if away_scoring > 0:
                reasons.append(f"Away team missing key scorers")

        elif market in ['over25', 'under25']:
            # Scoring vs defensive balance
            home_scoring = home_analysis.missing_impact.get('scoring', 0)
            away_scoring = away_analysis.missing_impact.get('scoring', 0)
            home_defensive = home_analysis.missing_impact.get('defensive', 0)
            away_defensive = away_analysis.missing_impact.get('defensive', 0)

            if market == 'over25':
                # Missing defenders = more goals, missing scorers = fewer
                total_impact = (home_defensive + away_defensive - home_scoring - away_scoring) * 0.3
            else:  # under25
                total_impact = (home_scoring + away_scoring - home_defensive - away_defensive) * 0.3

            if home_defensive > 0 or away_defensive > 0:
                reasons.append("Defensive absences may lead to goals")

        else:
            total_impact = 0

        # Apply adjustment with caps
        adjustment_factor = 1.0 + np.clip(total_impact, -self.max_adjustment, self.max_adjustment)

        # Calculate adjusted probability
        adjusted_prob = original_prob * adjustment_factor
        adjusted_prob = np.clip(adjusted_prob, 0.01, 0.99)

        # Determine confidence change direction
        if adjustment_factor > 1.02:
            confidence_change = 'boost'
        elif adjustment_factor < 0.98:
            confidence_change = 'reduce'
        else:
            confidence_change = 'neutral'

        if not reasons:
            reasons.append("No significant lineup changes detected")

        return ConfidenceAdjustment(
            fixture_id=fixture_id,
            market=market,
            original_probability=original_prob,
            adjusted_probability=adjusted_prob,
            adjustment_factor=adjustment_factor,
            confidence_change=confidence_change,
            reasons=reasons,
        )

    def _calculate_team_impact(
        self,
        analysis: LineupAnalysis,
        market_impacts: Dict[str, float],
        positive: bool = True,
    ) -> float:
        """Calculate team's impact on the market based on missing players."""
        total_impact = 0.0

        for category, missing_ratio in analysis.missing_impact.items():
            market_weight = market_impacts.get(category, 0.5)
            category_impact = missing_ratio * market_weight

            if positive:
                # Missing players hurt probability
                total_impact -= category_impact
            else:
                # Opponent missing players help probability
                total_impact += category_impact * 0.5  # Reduced effect

        return total_impact

    def adjust_predictions(
        self,
        predictions: pd.DataFrame,
        lineups: Dict[int, Dict[str, Any]],
        markets: List[str] = None,
    ) -> pd.DataFrame:
        """
        Adjust a DataFrame of predictions based on lineup data.

        Args:
            predictions: DataFrame with fixture_id, home_team_id, away_team_id,
                        and probability columns for each market
            lineups: Dict mapping fixture_id to lineup data
                    (contains 'home' and 'away' lineup dicts)
            markets: List of markets to adjust (default: all available)

        Returns:
            DataFrame with adjusted probabilities and adjustment info
        """
        if markets is None:
            markets = ['home_win', 'away_win', 'btts', 'over25']

        results = []

        for _, row in predictions.iterrows():
            fixture_id = row['fixture_id']
            home_team_id = row.get('home_team_id')
            away_team_id = row.get('away_team_id')
            home_team_name = row.get('home_team_name', f'Team {home_team_id}')
            away_team_name = row.get('away_team_name', f'Team {away_team_id}')

            # Get lineup data
            fixture_lineups = lineups.get(fixture_id, {})
            home_lineup = fixture_lineups.get('home', {})
            away_lineup = fixture_lineups.get('away', {})

            # Analyze lineups
            home_analysis = self.analyze_lineup(
                home_team_id, home_team_name, home_lineup)
            away_analysis = self.analyze_lineup(
                away_team_id, away_team_name, away_lineup)

            # Adjust each market
            row_data = row.to_dict()
            for market in markets:
                prob_col = f'{market}_prob'
                if prob_col not in row:
                    continue

                original_prob = row[prob_col]
                adjustment = self.calculate_adjustment(
                    fixture_id, market, original_prob,
                    home_analysis, away_analysis
                )

                row_data[f'{market}_prob_adj'] = adjustment.adjusted_probability
                row_data[f'{market}_adj_factor'] = adjustment.adjustment_factor
                row_data[f'{market}_adj_change'] = adjustment.confidence_change

            # Add lineup analysis summary
            row_data['home_strength'] = home_analysis.strength_score
            row_data['away_strength'] = away_analysis.strength_score
            row_data['home_missing_key'] = len(home_analysis.key_players_missing)
            row_data['away_missing_key'] = len(away_analysis.key_players_missing)

            results.append(row_data)

        return pd.DataFrame(results)


def adjust_single_prediction(
    fixture_id: int,
    market: str,
    original_prob: float,
    home_team_id: int,
    away_team_id: int,
    home_lineup: Dict[str, Any],
    away_lineup: Dict[str, Any],
    home_team_name: str = '',
    away_team_name: str = '',
) -> ConfidenceAdjustment:
    """
    Convenience function to adjust a single prediction.

    Args:
        fixture_id: Fixture ID
        market: Betting market
        original_prob: Original probability
        home_team_id: Home team ID
        away_team_id: Away team ID
        home_lineup: Home team lineup data
        away_lineup: Away team lineup data
        home_team_name: Home team name (optional)
        away_team_name: Away team name (optional)

    Returns:
        ConfidenceAdjustment with adjusted probability
    """
    adjuster = LineupConfidenceAdjuster()

    home_analysis = adjuster.analyze_lineup(
        home_team_id, home_team_name, home_lineup)
    away_analysis = adjuster.analyze_lineup(
        away_team_id, away_team_name, away_lineup)

    return adjuster.calculate_adjustment(
        fixture_id, market, original_prob,
        home_analysis, away_analysis
    )


if __name__ == "__main__":
    # Test the adjuster
    adjuster = LineupConfidenceAdjuster()

    # Simulated lineup (Liverpool full strength)
    liverpool_lineup = {
        'formation': '4-3-3',
        'startXI': [
            {'player': {'name': 'Alisson'}},
            {'player': {'name': 'Trent Alexander-Arnold'}},
            {'player': {'name': 'Virgil van Dijk'}},
            {'player': {'name': 'Ibrahima Konate'}},
            {'player': {'name': 'Andrew Robertson'}},
            {'player': {'name': 'Dominik Szoboszlai'}},
            {'player': {'name': 'Alexis Mac Allister'}},
            {'player': {'name': 'Curtis Jones'}},
            {'player': {'name': 'Mohamed Salah'}},
            {'player': {'name': 'Darwin Nunez'}},
            {'player': {'name': 'Luis Diaz'}},
        ]
    }

    # Simulated lineup (Man City with Haaland, KDB missing)
    city_lineup = {
        'formation': '4-3-3',
        'startXI': [
            {'player': {'name': 'Ederson'}},
            {'player': {'name': 'Kyle Walker'}},
            {'player': {'name': 'Ruben Dias'}},
            {'player': {'name': 'John Stones'}},
            {'player': {'name': 'Josko Gvardiol'}},
            {'player': {'name': 'Rodri'}},
            {'player': {'name': 'Mateo Kovacic'}},
            {'player': {'name': 'Bernardo Silva'}},
            {'player': {'name': 'Jack Grealish'}},
            {'player': {'name': 'Phil Foden'}},
            {'player': {'name': 'Julian Alvarez'}},  # No Haaland, no KDB
        ]
    }

    # Analyze lineups
    lfc_analysis = adjuster.analyze_lineup(40, 'Liverpool', liverpool_lineup)
    city_analysis = adjuster.analyze_lineup(50, 'Man City', city_lineup)

    print("Liverpool Analysis:")
    print(f"  Strength: {lfc_analysis.strength_score:.2f}")
    print(f"  Available: {lfc_analysis.key_players_available}")
    print(f"  Missing: {lfc_analysis.key_players_missing}")

    print("\nMan City Analysis:")
    print(f"  Strength: {city_analysis.strength_score:.2f}")
    print(f"  Available: {city_analysis.key_players_available}")
    print(f"  Missing: {city_analysis.key_players_missing}")
    print(f"  Missing Impact: {city_analysis.missing_impact}")

    # Test adjustment for home_win market
    # Liverpool home vs weakened City
    adjustment = adjuster.calculate_adjustment(
        fixture_id=12345,
        market='home_win',
        original_prob=0.45,
        home_analysis=lfc_analysis,
        away_analysis=city_analysis,
    )

    print(f"\nHome Win Adjustment (Liverpool vs City):")
    print(f"  Original: {adjustment.original_probability:.3f}")
    print(f"  Adjusted: {adjustment.adjusted_probability:.3f}")
    print(f"  Factor: {adjustment.adjustment_factor:.3f}")
    print(f"  Change: {adjustment.confidence_change}")
    print(f"  Reasons: {adjustment.reasons}")
