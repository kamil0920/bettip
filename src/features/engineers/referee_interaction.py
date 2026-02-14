"""Feature engineering - Referee × Team interaction features.

Current gap: Referee features are league-average (ref_cards_avg, ref_fouls_avg).
But a strict referee with a disciplined team produces fewer cards than the same
referee with an aggressive team.

These interaction features capture how specific referee-team pairings behave,
which is critical for fouls and cards market prediction.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.engineers.base import BaseFeatureEngineer


class RefereeTeamInteractionEngineer(BaseFeatureEngineer):
    """
    Creates referee × team interaction features.

    Captures specific referee-team pairing tendencies:
    - ref_team_cards_history: cards given by this ref to this team historically
    - ref_team_fouls_history: fouls called for this team by this referee
    - ref_strictness_vs_team_discipline: interaction of ref strictness × team fouls tendency
    - ref_home_bias_for_team: does this ref favor this specific home team?

    Walk-forward safe: only uses past matches with same referee-team pairing.
    """

    def __init__(self, min_encounters: int = 2):
        """
        Args:
            min_encounters: Minimum referee-team encounters to use specific history.
                           Falls back to league averages below this threshold.
        """
        self.min_encounters = min_encounters

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate referee-team interaction features."""
        matches = data['matches'].copy()
        matches = matches.sort_values('date').reset_index(drop=True)

        # Track referee-team history: {(referee, team_id): {cards, fouls, matches, home_wins}}
        ref_team_history: Dict[Tuple[str, int], Dict] = {}
        # Track team discipline (for interaction with ref strictness)
        team_fouls_ema: Dict[int, float] = {}
        # Track referee strictness
        ref_cards_avg: Dict[str, List[float]] = {}

        alpha = 2 / (10 + 1)  # EMA span=10

        features_list = []

        for _, match in matches.iterrows():
            referee = match.get('referee')
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            fid = match['fixture_id']

            features = {'fixture_id': fid}

            has_referee = pd.notna(referee) and isinstance(referee, str) and referee.strip()

            for prefix, team_id in [('home', home_id), ('away', away_id)]:
                if has_referee:
                    key = (referee, team_id)
                    hist = ref_team_history.get(key)

                    if hist and hist['matches'] >= self.min_encounters:
                        n = hist['matches']
                        features[f'{prefix}_ref_team_cards_avg'] = hist['cards'] / n
                        features[f'{prefix}_ref_team_fouls_avg'] = hist['fouls'] / n
                    else:
                        features[f'{prefix}_ref_team_cards_avg'] = 0.0
                        features[f'{prefix}_ref_team_fouls_avg'] = 0.0

                    # Referee strictness × team discipline interaction
                    ref_strictness = np.mean(ref_cards_avg.get(referee, [4.2]))
                    team_discipline = team_fouls_ema.get(team_id, 11.0)  # avg fouls per team
                    features[f'{prefix}_ref_strictness_x_discipline'] = (
                        ref_strictness * team_discipline / 46.2  # normalize by avg (4.2 * 11.0)
                    )

                    # Ref home bias for this specific team
                    if prefix == 'home' and hist and hist['matches'] >= self.min_encounters:
                        features['ref_home_bias_for_team'] = (
                            hist.get('home_wins', 0) / hist['matches']
                        )
                    elif prefix == 'home':
                        features['ref_home_bias_for_team'] = 0.46  # league avg
                else:
                    features[f'{prefix}_ref_team_cards_avg'] = 0.0
                    features[f'{prefix}_ref_team_fouls_avg'] = 0.0
                    features[f'{prefix}_ref_strictness_x_discipline'] = 1.0
                    if prefix == 'home':
                        features['ref_home_bias_for_team'] = 0.46

            features_list.append(features)

            # Update histories AFTER recording features (walk-forward safe)
            if has_referee:
                home_goals = self._safe_val(match, ['ft_home', 'home_goals', 'FTHG'])
                away_goals = self._safe_val(match, ['ft_away', 'away_goals', 'FTAG'])
                home_yellows = self._safe_val(match, ['home_yellow_cards', 'home_yellows', 'HY'])
                away_yellows = self._safe_val(match, ['away_yellow_cards', 'away_yellows', 'AY'])
                home_reds = self._safe_val(match, ['home_red_cards', 'home_reds', 'HR'])
                away_reds = self._safe_val(match, ['away_red_cards', 'away_reds', 'AR'])
                home_fouls = self._safe_val(match, ['home_fouls', 'HF'])
                away_fouls = self._safe_val(match, ['away_fouls', 'AF'])

                total_cards = home_yellows + away_yellows + home_reds + away_reds

                # Update ref-team pairs
                for tid, cards, fouls, is_home in [
                    (home_id, home_yellows + home_reds, home_fouls, True),
                    (away_id, away_yellows + away_reds, away_fouls, False),
                ]:
                    key = (referee, tid)
                    if key not in ref_team_history:
                        ref_team_history[key] = {
                            'cards': 0, 'fouls': 0, 'matches': 0, 'home_wins': 0
                        }
                    ref_team_history[key]['cards'] += cards
                    ref_team_history[key]['fouls'] += fouls
                    ref_team_history[key]['matches'] += 1
                    if is_home and home_goals > away_goals:
                        ref_team_history[key]['home_wins'] += 1

                # Update referee strictness tracker
                if referee not in ref_cards_avg:
                    ref_cards_avg[referee] = []
                ref_cards_avg[referee].append(total_cards)

                # Update team fouls EMA
                for tid, fouls in [(home_id, home_fouls), (away_id, away_fouls)]:
                    if tid not in team_fouls_ema:
                        team_fouls_ema[tid] = fouls
                    else:
                        team_fouls_ema[tid] = alpha * fouls + (1 - alpha) * team_fouls_ema[tid]

        print(f"Created {len(features_list)} referee-team interaction features")
        return pd.DataFrame(features_list)

    def _safe_val(self, match: pd.Series, keys: List[str], default: float = 0.0) -> float:
        """Safely get a value from match, trying multiple column names."""
        for key in keys:
            if key in match.index:
                val = match[key]
                if pd.notna(val):
                    return float(val)
        return default
