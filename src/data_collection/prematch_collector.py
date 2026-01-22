"""
Pre-match data collector for betting intelligence.

Collects:
1. Injuries - Players unavailable for upcoming fixtures
2. Lineups - Confirmed starting XI (available ~1hr before kickoff)
3. Predictions - API predictions, form comparison, team stats

Usage:
    collector = PreMatchCollector()

    # Get all pre-match data for upcoming fixtures
    data = collector.collect_prematch_data(league_id=39, season=2024)

    # Get injuries for specific fixture
    injuries = collector.get_injuries(fixture_id=1234567)

    # Get lineups (when available)
    lineups = collector.get_lineups(fixture_id=1234567)
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from src.data_collection.api_client import FootballAPIClient

logger = logging.getLogger(__name__)


# League IDs for supported leagues
LEAGUE_IDS = {
    'premier_league': 39,
    'la_liga': 140,
    'serie_a': 135,
    'bundesliga': 78,
    'ligue_1': 61,
}


class PreMatchCollector:
    """
    Collector for pre-match betting intelligence data.

    Provides structured access to:
    - Player injuries and availability
    - Confirmed lineups and formations
    - API predictions and team comparisons
    - Head-to-head statistics
    """

    def __init__(self, client: Optional[FootballAPIClient] = None):
        """
        Initialize collector.

        Args:
            client: Optional API client. Creates new one if not provided.
        """
        self.client = client or FootballAPIClient()
        self.logger = logging.getLogger(self.__class__.__name__)

    # =========================================================================
    # INJURIES
    # =========================================================================

    def get_injuries_by_league(
        self,
        league_id: int,
        season: int
    ) -> pd.DataFrame:
        """
        Get all injuries for a league/season.

        Args:
            league_id: API-Football league ID (e.g., 39 for Premier League)
            season: Season year (e.g., 2024)

        Returns:
            DataFrame with columns:
                - fixture_id, fixture_date
                - team_id, team_name
                - player_id, player_name
                - injury_type, injury_reason
        """
        self.logger.info(f"Fetching injuries for league {league_id}, season {season}")

        try:
            response = self.client._make_request(
                '/injuries',
                {'league': league_id, 'season': season}
            )

            injuries = response.get('response', [])
            self.logger.info(f"Found {len(injuries)} injury records")

            if not injuries:
                return pd.DataFrame()

            records = []
            for inj in injuries:
                records.append({
                    'fixture_id': inj['fixture']['id'],
                    'fixture_date': inj['fixture']['date'],
                    'team_id': inj['team']['id'],
                    'team_name': inj['team']['name'],
                    'player_id': inj['player']['id'],
                    'player_name': inj['player']['name'],
                    'injury_type': inj['player'].get('type', 'Unknown'),
                    'injury_reason': inj['player'].get('reason', 'Unknown'),
                })

            df = pd.DataFrame(records)
            df['fixture_date'] = pd.to_datetime(df['fixture_date'])
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch injuries: {e}")
            return pd.DataFrame()

    def get_injuries_by_fixture(self, fixture_id: int) -> pd.DataFrame:
        """
        Get injuries for a specific fixture.

        Args:
            fixture_id: The fixture ID

        Returns:
            DataFrame with injury details
        """
        try:
            response = self.client._make_request(
                '/injuries',
                {'fixture': fixture_id}
            )

            injuries = response.get('response', [])

            if not injuries:
                return pd.DataFrame()

            records = []
            for inj in injuries:
                records.append({
                    'fixture_id': fixture_id,
                    'team_id': inj['team']['id'],
                    'team_name': inj['team']['name'],
                    'player_id': inj['player']['id'],
                    'player_name': inj['player']['name'],
                    'injury_type': inj['player'].get('type', 'Unknown'),
                    'injury_reason': inj['player'].get('reason', 'Unknown'),
                })

            return pd.DataFrame(records)

        except Exception as e:
            self.logger.error(f"Failed to fetch injuries for fixture {fixture_id}: {e}")
            return pd.DataFrame()

    def get_injuries_by_team(self, team_id: int) -> pd.DataFrame:
        """
        Get current injuries for a specific team.

        Args:
            team_id: The team ID

        Returns:
            DataFrame with current team injuries
        """
        try:
            response = self.client._make_request(
                '/injuries',
                {'team': team_id}
            )

            injuries = response.get('response', [])

            if not injuries:
                return pd.DataFrame()

            records = []
            for inj in injuries:
                records.append({
                    'fixture_id': inj['fixture']['id'],
                    'fixture_date': inj['fixture']['date'],
                    'team_id': team_id,
                    'player_id': inj['player']['id'],
                    'player_name': inj['player']['name'],
                    'injury_type': inj['player'].get('type', 'Unknown'),
                    'injury_reason': inj['player'].get('reason', 'Unknown'),
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df['fixture_date'] = pd.to_datetime(df['fixture_date'])
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch injuries for team {team_id}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # LINEUPS
    # =========================================================================

    def get_lineups(self, fixture_id: int) -> Dict[str, Any]:
        """
        Get confirmed lineups for a fixture.

        Note: Lineups typically available ~1 hour before kickoff.

        Args:
            fixture_id: The fixture ID

        Returns:
            Dict with structure:
            {
                'home': {
                    'team_id': int,
                    'team_name': str,
                    'formation': str,
                    'coach': str,
                    'starting_xi': [{'id': int, 'name': str, 'number': int, 'pos': str}],
                    'substitutes': [...]
                },
                'away': {...}
            }
        """
        try:
            response = self.client._make_request(
                '/fixtures/lineups',
                {'fixture': fixture_id}
            )

            lineups = response.get('response', [])

            if not lineups:
                return {'home': None, 'away': None, 'available': False}

            result = {'available': True}

            for lineup in lineups:
                team_id = lineup['team']['id']
                team_name = lineup['team']['name']

                # Determine home/away based on position in response
                side = 'home' if lineups.index(lineup) == 0 else 'away'

                starting_xi = []
                for player in lineup.get('startXI', []):
                    p = player['player']
                    starting_xi.append({
                        'id': p['id'],
                        'name': p['name'],
                        'number': p.get('number'),
                        'position': p.get('pos'),
                        'grid': p.get('grid'),
                    })

                substitutes = []
                for player in lineup.get('substitutes', []):
                    p = player['player']
                    substitutes.append({
                        'id': p['id'],
                        'name': p['name'],
                        'number': p.get('number'),
                        'position': p.get('pos'),
                    })

                result[side] = {
                    'team_id': team_id,
                    'team_name': team_name,
                    'formation': lineup.get('formation'),
                    'coach': lineup.get('coach', {}).get('name'),
                    'starting_xi': starting_xi,
                    'substitutes': substitutes,
                }

            return result

        except Exception as e:
            self.logger.error(f"Failed to fetch lineups for fixture {fixture_id}: {e}")
            return {'home': None, 'away': None, 'available': False}

    def get_lineups_flat(self, fixture_id: int) -> pd.DataFrame:
        """
        Get lineups as a flat DataFrame.

        Args:
            fixture_id: The fixture ID

        Returns:
            DataFrame with columns:
                - fixture_id, team_id, team_name
                - player_id, player_name, position, number
                - is_starter, formation
        """
        lineups = self.get_lineups(fixture_id)

        if not lineups.get('available'):
            return pd.DataFrame()

        records = []
        for side in ['home', 'away']:
            if not lineups.get(side):
                continue

            team = lineups[side]

            for player in team.get('starting_xi', []):
                records.append({
                    'fixture_id': fixture_id,
                    'team_id': team['team_id'],
                    'team_name': team['team_name'],
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'position': player['position'],
                    'number': player['number'],
                    'is_starter': True,
                    'formation': team['formation'],
                    'coach': team['coach'],
                })

            for player in team.get('substitutes', []):
                records.append({
                    'fixture_id': fixture_id,
                    'team_id': team['team_id'],
                    'team_name': team['team_name'],
                    'player_id': player['id'],
                    'player_name': player['name'],
                    'position': player['position'],
                    'number': player['number'],
                    'is_starter': False,
                    'formation': team['formation'],
                    'coach': team['coach'],
                })

        return pd.DataFrame(records)

    # =========================================================================
    # PREDICTIONS & INTELLIGENCE
    # =========================================================================

    def get_predictions(self, fixture_id: int) -> Dict[str, Any]:
        """
        Get API predictions and comparison data for a fixture.

        Args:
            fixture_id: The fixture ID

        Returns:
            Dict with structure:
            {
                'winner': {'id': int, 'name': str, 'comment': str},
                'percent': {'home': '50%', 'draw': '25%', 'away': '25%'},
                'advice': str,
                'goals': {'home': str, 'away': str},
                'comparison': {
                    'form': {'home': '60%', 'away': '40%'},
                    'att': {...},
                    'def': {...},
                    'poisson_distribution': {...},
                    'h2h': {...},
                    'goals': {...},
                    'total': {...}
                },
                'teams': {
                    'home': {'form': str, 'goals_for': {...}, 'goals_against': {...}},
                    'away': {...}
                }
            }
        """
        try:
            response = self.client._make_request(
                '/predictions',
                {'fixture': fixture_id}
            )

            predictions = response.get('response', [])

            if not predictions:
                return {}

            pred = predictions[0]

            result = {
                'fixture_id': fixture_id,
                'winner': pred.get('predictions', {}).get('winner'),
                'percent': pred.get('predictions', {}).get('percent'),
                'advice': pred.get('predictions', {}).get('advice'),
                'goals': pred.get('predictions', {}).get('goals'),
                'win_or_draw': pred.get('predictions', {}).get('win_or_draw'),
                'under_over': pred.get('predictions', {}).get('under_over'),
                'comparison': pred.get('comparison', {}),
                'teams': {},
            }

            # Extract team details
            for side in ['home', 'away']:
                team = pred.get('teams', {}).get(side, {})
                league = team.get('league', {})

                result['teams'][side] = {
                    'id': team.get('id'),
                    'name': team.get('name'),
                    'form': league.get('form'),
                    'fixtures': league.get('fixtures', {}),
                    'goals_for': league.get('goals', {}).get('for', {}),
                    'goals_against': league.get('goals', {}).get('against', {}),
                }

            return result

        except Exception as e:
            self.logger.error(f"Failed to fetch predictions for fixture {fixture_id}: {e}")
            return {}

    def get_predictions_flat(self, fixture_id: int) -> pd.DataFrame:
        """
        Get predictions as a flat DataFrame row.

        Args:
            fixture_id: The fixture ID

        Returns:
            DataFrame with one row containing all prediction features
        """
        pred = self.get_predictions(fixture_id)

        if not pred:
            return pd.DataFrame()

        def parse_pct(pct_str: str) -> float:
            """Convert '50%' to 0.50"""
            if not pct_str:
                return 0.0
            try:
                return float(pct_str.replace('%', '')) / 100
            except:
                return 0.0

        record = {
            'fixture_id': fixture_id,
            # Winner prediction
            'pred_winner_id': pred.get('winner', {}).get('id') if pred.get('winner') else None,
            'pred_winner_name': pred.get('winner', {}).get('name') if pred.get('winner') else None,
            'pred_advice': pred.get('advice'),
            'pred_win_or_draw': pred.get('win_or_draw'),
            # Percentages
            'pred_home_pct': parse_pct(pred.get('percent', {}).get('home', '0%')),
            'pred_draw_pct': parse_pct(pred.get('percent', {}).get('draw', '0%')),
            'pred_away_pct': parse_pct(pred.get('percent', {}).get('away', '0%')),
        }

        # Comparison metrics
        comparison = pred.get('comparison', {})
        for metric in ['form', 'att', 'def', 'poisson_distribution', 'h2h', 'goals', 'total']:
            if metric in comparison:
                record[f'comp_{metric}_home'] = parse_pct(comparison[metric].get('home', '0%'))
                record[f'comp_{metric}_away'] = parse_pct(comparison[metric].get('away', '0%'))

        # Team form
        for side in ['home', 'away']:
            team = pred.get('teams', {}).get(side, {})
            form = team.get('form', '')

            # Calculate recent form metrics
            last_5 = form[-5:] if form else ''
            record[f'{side}_form_last5'] = last_5
            record[f'{side}_form_wins'] = last_5.count('W')
            record[f'{side}_form_draws'] = last_5.count('D')
            record[f'{side}_form_losses'] = last_5.count('L')
            record[f'{side}_form_points'] = last_5.count('W') * 3 + last_5.count('D')

            # Goals
            goals_for = team.get('goals_for', {})
            goals_against = team.get('goals_against', {})

            record[f'{side}_goals_for_total'] = goals_for.get('total', {}).get('total', 0)
            record[f'{side}_goals_for_avg'] = float(goals_for.get('average', {}).get('total', '0'))
            record[f'{side}_goals_against_total'] = goals_against.get('total', {}).get('total', 0)
            record[f'{side}_goals_against_avg'] = float(goals_against.get('average', {}).get('total', '0'))

            # Under/Over history
            for line in ['0.5', '1.5', '2.5', '3.5']:
                uo = goals_for.get('under_over', {}).get(line, {})
                record[f'{side}_goals_over_{line.replace(".", "_")}'] = uo.get('over', 0)
                record[f'{side}_goals_under_{line.replace(".", "_")}'] = uo.get('under', 0)

        return pd.DataFrame([record])

    # =========================================================================
    # HEAD TO HEAD
    # =========================================================================

    def get_h2h(self, team1_id: int, team2_id: int, last: int = 10) -> pd.DataFrame:
        """
        Get head-to-head history between two teams.

        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last: Number of recent matches to fetch

        Returns:
            DataFrame with H2H match history
        """
        try:
            response = self.client._make_request(
                '/fixtures/headtohead',
                {'h2h': f'{team1_id}-{team2_id}', 'last': last}
            )

            matches = response.get('response', [])

            if not matches:
                return pd.DataFrame()

            records = []
            for match in matches:
                records.append({
                    'fixture_id': match['fixture']['id'],
                    'date': match['fixture']['date'],
                    'home_team_id': match['teams']['home']['id'],
                    'home_team_name': match['teams']['home']['name'],
                    'away_team_id': match['teams']['away']['id'],
                    'away_team_name': match['teams']['away']['name'],
                    'home_goals': match['goals']['home'],
                    'away_goals': match['goals']['away'],
                    'home_winner': match['teams']['home'].get('winner'),
                    'away_winner': match['teams']['away'].get('winner'),
                })

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch H2H for {team1_id} vs {team2_id}: {e}")
            return pd.DataFrame()

    def get_h2h_summary(self, team1_id: int, team2_id: int, last: int = 10) -> Dict[str, Any]:
        """
        Get summarized H2H statistics.

        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last: Number of recent matches to analyze

        Returns:
            Dict with H2H summary stats
        """
        h2h = self.get_h2h(team1_id, team2_id, last)

        if h2h.empty:
            return {}

        # Calculate stats for team1
        team1_home = h2h[h2h['home_team_id'] == team1_id]
        team1_away = h2h[h2h['away_team_id'] == team1_id]

        team1_wins = (
            (team1_home['home_winner'] == True).sum() +
            (team1_away['away_winner'] == True).sum()
        )
        team2_wins = len(h2h) - team1_wins - (
            (h2h['home_winner'].isna()) | (h2h['home_winner'] == False) & (h2h['away_winner'] == False)
        ).sum()
        draws = len(h2h) - team1_wins - team2_wins

        team1_goals = (
            team1_home['home_goals'].sum() +
            team1_away['away_goals'].sum()
        )
        team2_goals = (
            team1_home['away_goals'].sum() +
            team1_away['home_goals'].sum()
        )

        return {
            'matches': len(h2h),
            'team1_id': team1_id,
            'team2_id': team2_id,
            'team1_wins': int(team1_wins),
            'team2_wins': int(team2_wins),
            'draws': int(draws),
            'team1_goals': int(team1_goals),
            'team2_goals': int(team2_goals),
            'team1_win_pct': team1_wins / len(h2h) if len(h2h) > 0 else 0,
            'avg_total_goals': (team1_goals + team2_goals) / len(h2h) if len(h2h) > 0 else 0,
        }

    # =========================================================================
    # UPCOMING FIXTURES
    # =========================================================================

    def get_upcoming_fixtures(
        self,
        league_id: int,
        season: int,
        next_n: int = 10
    ) -> pd.DataFrame:
        """
        Get upcoming fixtures for a league.

        Args:
            league_id: API-Football league ID
            season: Season year
            next_n: Number of upcoming fixtures to fetch

        Returns:
            DataFrame with upcoming fixtures
        """
        try:
            response = self.client._make_request(
                '/fixtures',
                {'league': league_id, 'season': season, 'next': next_n}
            )

            fixtures = response.get('response', [])

            if not fixtures:
                return pd.DataFrame()

            records = []
            for f in fixtures:
                records.append({
                    'fixture_id': f['fixture']['id'],
                    'date': f['fixture']['date'],
                    'timestamp': f['fixture']['timestamp'],
                    'venue_name': f['fixture'].get('venue', {}).get('name'),
                    'venue_city': f['fixture'].get('venue', {}).get('city'),
                    'status': f['fixture'].get('status', {}).get('short'),
                    'home_team_id': f['teams']['home']['id'],
                    'home_team_name': f['teams']['home']['name'],
                    'away_team_id': f['teams']['away']['id'],
                    'away_team_name': f['teams']['away']['name'],
                    'league_id': f['league']['id'],
                    'league_name': f['league']['name'],
                    'round': f['league'].get('round'),
                })

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch upcoming fixtures: {e}")
            return pd.DataFrame()

    # =========================================================================
    # COMBINED PRE-MATCH DATA
    # =========================================================================

    def collect_prematch_data(
        self,
        fixture_id: int,
        include_h2h: bool = True
    ) -> Dict[str, Any]:
        """
        Collect all pre-match data for a single fixture.

        Args:
            fixture_id: The fixture ID
            include_h2h: Whether to fetch H2H data

        Returns:
            Dict with all pre-match data:
            {
                'fixture_id': int,
                'injuries': DataFrame,
                'lineups': Dict,
                'predictions': Dict,
                'h2h': DataFrame (optional)
            }
        """
        self.logger.info(f"Collecting pre-match data for fixture {fixture_id}")

        result = {
            'fixture_id': fixture_id,
            'collected_at': datetime.utcnow().isoformat(),
        }

        # Get injuries
        result['injuries'] = self.get_injuries_by_fixture(fixture_id)
        self.logger.info(f"Found {len(result['injuries'])} injuries")

        # Get lineups
        result['lineups'] = self.get_lineups(fixture_id)
        self.logger.info(f"Lineups available: {result['lineups'].get('available', False)}")

        # Get predictions
        result['predictions'] = self.get_predictions(fixture_id)
        self.logger.info(f"Predictions fetched: {bool(result['predictions'])}")

        # Get H2H if requested and we have team IDs from predictions
        if include_h2h and result['predictions']:
            teams = result['predictions'].get('teams', {})
            home_id = teams.get('home', {}).get('id')
            away_id = teams.get('away', {}).get('id')

            if home_id and away_id:
                result['h2h'] = self.get_h2h(home_id, away_id)
                result['h2h_summary'] = self.get_h2h_summary(home_id, away_id)
                self.logger.info(f"H2H matches: {len(result['h2h'])}")

        return result

    def collect_all_upcoming(
        self,
        league_id: int,
        season: int,
        next_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Collect pre-match data for all upcoming fixtures in a league.

        Args:
            league_id: API-Football league ID
            season: Season year
            next_n: Number of upcoming fixtures

        Returns:
            List of pre-match data dicts
        """
        fixtures = self.get_upcoming_fixtures(league_id, season, next_n)

        if fixtures.empty:
            self.logger.warning("No upcoming fixtures found")
            return []

        results = []
        for _, fixture in fixtures.iterrows():
            data = self.collect_prematch_data(fixture['fixture_id'])
            data['fixture_info'] = fixture.to_dict()
            results.append(data)

        return results


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Collect pre-match data')
    parser.add_argument('--league', type=str, default='premier_league',
                       choices=list(LEAGUE_IDS.keys()))
    parser.add_argument('--season', type=int, default=2024)
    parser.add_argument('--fixture', type=int, help='Specific fixture ID')
    parser.add_argument('--upcoming', type=int, default=5, help='Number of upcoming fixtures')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    collector = PreMatchCollector()
    league_id = LEAGUE_IDS[args.league]

    if args.fixture:
        # Collect for specific fixture
        data = collector.collect_prematch_data(args.fixture)

        print(f"\n=== Fixture {args.fixture} ===")
        print(f"Injuries: {len(data['injuries'])}")
        print(f"Lineups available: {data['lineups'].get('available')}")

        if data['predictions']:
            pred = data['predictions']
            print(f"Prediction: {pred.get('advice')}")
            print(f"Win %: Home={pred.get('percent', {}).get('home')}, "
                  f"Draw={pred.get('percent', {}).get('draw')}, "
                  f"Away={pred.get('percent', {}).get('away')}")
    else:
        # Collect for upcoming fixtures
        print(f"\n=== Upcoming {args.league.upper()} Fixtures ===")
        fixtures = collector.get_upcoming_fixtures(league_id, args.season, args.upcoming)

        if fixtures.empty:
            print("No upcoming fixtures found")
        else:
            for _, f in fixtures.iterrows():
                print(f"{f['date'].strftime('%Y-%m-%d %H:%M')} | "
                      f"{f['home_team_name']} vs {f['away_team_name']}")

            # Collect full data for first fixture
            if not fixtures.empty:
                first_id = fixtures.iloc[0]['fixture_id']
                print(f"\n=== Data for first fixture ({first_id}) ===")
                data = collector.collect_prematch_data(first_id)

                if args.output:
                    # Convert DataFrames to dicts for JSON serialization
                    output_data = {
                        'fixture_id': data['fixture_id'],
                        'collected_at': data['collected_at'],
                        'injuries': data['injuries'].to_dict('records') if not data['injuries'].empty else [],
                        'lineups': data['lineups'],
                        'predictions': data['predictions'],
                        'h2h_summary': data.get('h2h_summary', {}),
                    }
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=2, default=str)
                    print(f"Saved to {args.output}")
