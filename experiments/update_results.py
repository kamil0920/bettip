#!/usr/bin/env python3
"""
Auto-update betting results from API-Football.

Fetches actual match statistics (fouls, shots, corners, goals) and updates:
- data/paper_trading/paper_trades.csv
- data/05-recommendations/rec_*.csv

Usage:
    python experiments/update_results.py              # Update all pending bets
    python experiments/update_results.py --dry-run    # Preview without saving
    python experiments/update_results.py --file rec_20260122_001.csv  # Specific file
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.api_client import FootballAPIClient
from src.data_collection.match_stats_collector import MatchStatsCollector

# Paths
PAPER_TRADES_FILE = project_root / 'data/paper_trading/paper_trades.csv'
RECOMMENDATIONS_DIR = project_root / 'data/05-recommendations'

# League mappings
LEAGUE_IDS = {
    'premier_league': 39,
    'la_liga': 140,
    'serie_a': 135,
    'bundesliga': 78,
    'ligue_1': 61,
}


class ResultsUpdater:
    """Auto-update betting results from API-Football."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.client = FootballAPIClient()
        self.stats_collector = MatchStatsCollector(data_dir=str(project_root / 'data/01-raw'))
        self._fixtures_cache: Dict[str, pd.DataFrame] = {}
        self._stats_cache: Dict[int, Dict] = {}

    def _load_fixtures(self, league: str, season: int = 2025) -> pd.DataFrame:
        """Load fixtures from local data or API."""
        cache_key = f"{league}_{season}"
        if cache_key in self._fixtures_cache:
            return self._fixtures_cache[cache_key]

        # Try loading from local parquet
        parquet_path = project_root / f'data/01-raw/{league}/{season}/matches.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self._fixtures_cache[cache_key] = df
            return df

        # Fallback to API
        league_id = LEAGUE_IDS.get(league)
        if league_id:
            fixtures = self.client.get_fixtures(league_id, season)
            df = pd.DataFrame(fixtures)
            self._fixtures_cache[cache_key] = df
            return df

        return pd.DataFrame()

    def _load_match_stats(self, league: str, season: int = 2025) -> pd.DataFrame:
        """Load match statistics from local data."""
        cache_key = f"{league}_{season}_stats"
        if cache_key in self._fixtures_cache:
            return self._fixtures_cache[cache_key]

        parquet_path = project_root / f'data/01-raw/{league}/{season}/match_stats.parquet'
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self._fixtures_cache[cache_key] = df
            return df

        return pd.DataFrame()

    def _find_fixture(self, home_team: str, away_team: str, match_date: str, league: str) -> Optional[Dict]:
        """Find fixture by team names and date. Returns fixture info with goals."""
        df = self._load_fixtures(league)
        if df.empty:
            return None

        # Normalize team names for matching
        def normalize(name: str) -> str:
            return name.lower().replace(' ', '').replace('-', '').replace('.', '')[:10]

        home_norm = normalize(home_team)
        away_norm = normalize(away_team)

        for _, row in df.iterrows():
            # Handle flattened parquet structure
            if 'teams.home.name' in row.index:
                row_home = str(row.get('teams.home.name', ''))
                row_away = str(row.get('teams.away.name', ''))
                fixture_date = str(row.get('fixture.date', ''))[:10]
                fixture_id = row.get('fixture.id')
                home_goals = row.get('goals.home')
                away_goals = row.get('goals.away')
                status = row.get('fixture.status.short', '')
            elif 'teams' in row:
                row_home = row['teams'].get('home', {}).get('name', '')
                row_away = row['teams'].get('away', {}).get('name', '')
                fixture_date = str(row.get('fixture', {}).get('date', ''))[:10]
                fixture_id = row.get('fixture', {}).get('id')
                home_goals = row.get('goals', {}).get('home')
                away_goals = row.get('goals', {}).get('away')
                status = row.get('fixture', {}).get('status', {}).get('short', '')
            else:
                row_home = str(row.get('home_team', ''))
                row_away = str(row.get('away_team', ''))
                fixture_date = str(row.get('date', ''))[:10]
                fixture_id = row.get('fixture_id')
                home_goals = row.get('home_goals')
                away_goals = row.get('away_goals')
                status = row.get('status', '')

            if normalize(row_home).startswith(home_norm) or home_norm in normalize(row_home):
                if normalize(row_away).startswith(away_norm) or away_norm in normalize(row_away):
                    if fixture_date == match_date[:10]:
                        return {
                            'fixture_id': fixture_id,
                            'home_goals': int(home_goals) if pd.notna(home_goals) else None,
                            'away_goals': int(away_goals) if pd.notna(away_goals) else None,
                            'status': status,
                            'home_team': row_home,
                            'away_team': row_away,
                        }

        return None

    def _get_match_stats(self, fixture_id: int, league: str = None) -> Optional[Dict]:
        """Get match statistics from local data or API."""
        if fixture_id in self._stats_cache:
            return self._stats_cache[fixture_id]

        # Try loading from local parquet first
        if league:
            stats_df = self._load_match_stats(league)
            if not stats_df.empty and 'fixture_id' in stats_df.columns:
                match_row = stats_df[stats_df['fixture_id'] == fixture_id]
                if not match_row.empty:
                    row = match_row.iloc[0]
                    stats = {
                        'home': {
                            'fouls': int(row.get('home_fouls', 0) or 0),
                            'shots': int(row.get('home_shots', 0) or 0),
                            'corner_kicks': int(row.get('home_corners', 0) or 0),
                            'shots_on_target': int(row.get('home_shots_on_target', 0) or 0),
                        },
                        'away': {
                            'fouls': int(row.get('away_fouls', 0) or 0),
                            'shots': int(row.get('away_shots', 0) or 0),
                            'corner_kicks': int(row.get('away_corners', 0) or 0),
                            'shots_on_target': int(row.get('away_shots_on_target', 0) or 0),
                        },
                        'home_goals': int(row.get('home_goals', 0) or 0),
                        'away_goals': int(row.get('away_goals', 0) or 0),
                    }
                    self._stats_cache[fixture_id] = stats
                    return stats

        # Fallback to API
        try:
            stats = self.stats_collector.collect_fixture_stats(fixture_id)
            if stats:
                self._stats_cache[fixture_id] = stats
                return stats
        except Exception as e:
            print(f"    ⚠ Error fetching stats for fixture {fixture_id}: {e}")

        return None

    def _evaluate_bet(self, bet_type: str, line: float, stats: Dict) -> Tuple[str, float]:
        """
        Evaluate if bet won or lost.

        Returns: (status, actual_value)
        """
        # Extract actual values from stats
        home_stats = stats.get('home', {})
        away_stats = stats.get('away', {})

        # Calculate totals
        total_fouls = (home_stats.get('fouls', 0) or 0) + (away_stats.get('fouls', 0) or 0)
        total_shots = (home_stats.get('shots', 0) or 0) + (away_stats.get('shots', 0) or 0)
        total_corners = (home_stats.get('corner_kicks', 0) or 0) + (away_stats.get('corner_kicks', 0) or 0)
        home_goals = stats.get('home_goals', 0) or 0
        away_goals = stats.get('away_goals', 0) or 0

        bet_lower = bet_type.lower()

        # FOULS
        if 'fouls' in bet_lower:
            actual = total_fouls
            if 'under' in bet_lower:
                status = 'WON' if actual < line else 'LOST'
            else:  # OVER
                status = 'WON' if actual > line else 'LOST'
            return status, actual

        # SHOTS
        if 'shots' in bet_lower:
            actual = total_shots
            if 'under' in bet_lower:
                status = 'WON' if actual < line else 'LOST'
            else:  # OVER
                status = 'WON' if actual > line else 'LOST'
            return status, actual

        # CORNERS
        if 'corner' in bet_lower:
            actual = total_corners
            if 'under' in bet_lower:
                status = 'WON' if actual < line else 'LOST'
            else:  # OVER
                status = 'WON' if actual > line else 'LOST'
            return status, actual

        # HOME WIN
        if 'home win' in bet_lower or bet_lower == 'home':
            actual = home_goals - away_goals
            status = 'WON' if home_goals > away_goals else 'LOST'
            return status, actual

        # AWAY WIN
        if 'away win' in bet_lower or bet_lower == 'away':
            actual = away_goals - home_goals
            status = 'WON' if away_goals > home_goals else 'LOST'
            return status, actual

        # ASIAN HANDICAP -0.5 (same as win)
        if '-0.5' in bet_lower:
            if 'home' in bet_lower:
                actual = home_goals - away_goals
                status = 'WON' if home_goals > away_goals else 'LOST'
            else:
                actual = away_goals - home_goals
                status = 'WON' if away_goals > home_goals else 'LOST'
            return status, actual

        return 'UNKNOWN', 0

    def _calculate_pnl(self, status: str, stake: float, odds: float) -> float:
        """Calculate profit/loss."""
        if status == 'WON':
            return stake * (odds - 1)
        elif status == 'LOST':
            return -stake
        return 0

    def update_paper_trades(self) -> Dict:
        """Update paper_trades.csv with results."""
        if not PAPER_TRADES_FILE.exists():
            print("Paper trades file not found")
            return {'updated': 0, 'errors': 0}

        df = pd.read_csv(PAPER_TRADES_FILE)
        pending = df[df['status'] == 'pending'].copy()

        today = datetime.now().date()
        updated = 0
        errors = 0

        print(f"\nUpdating paper trades ({len(pending)} pending)...")

        for idx, row in pending.iterrows():
            match_date = pd.to_datetime(row['date']).date()

            # Only update past matches
            if match_date >= today:
                continue

            print(f"\n  [{row['date']}] {row['home_team']} vs {row['away_team']}")
            print(f"    Bet: {row['bet_type']}")

            # Find fixture
            fixture_info = self._find_fixture(
                row['home_team'],
                row['away_team'],
                row['date'],
                row['league']
            )

            if not fixture_info:
                print(f"    ⚠ Fixture not found")
                errors += 1
                continue

            fixture_id = fixture_info['fixture_id']

            # Check if match is finished
            if fixture_info.get('status') not in ['FT', 'AET', 'PEN']:
                print(f"    ⚠ Match not finished yet (status: {fixture_info.get('status')})")
                continue

            # Get stats (fouls, shots, corners)
            stats = self._get_match_stats(fixture_id, row['league'])

            # Add goals from fixture info
            if stats is None:
                stats = {'home': {}, 'away': {}}
            stats['home_goals'] = fixture_info.get('home_goals', 0)
            stats['away_goals'] = fixture_info.get('away_goals', 0)

            # Evaluate bet
            line = 0
            if 'bet_type' in row and any(c.isdigit() for c in str(row['bet_type'])):
                # Extract line from bet_type like "Fouls UNDER 26.5"
                parts = str(row['bet_type']).split()
                for part in parts:
                    try:
                        line = float(part)
                        break
                    except ValueError:
                        continue

            status, actual = self._evaluate_bet(row['bet_type'], line, stats)
            pnl = self._calculate_pnl(status, row['stake'], row['odds'])

            print(f"    Actual: {actual}, Line: {line} → {status}")
            print(f"    P&L: ${pnl:+.2f}")

            if not self.dry_run:
                df.loc[idx, 'status'] = status.lower()
                df.loc[idx, 'result'] = 'win' if status == 'WON' else 'loss'
                df.loc[idx, 'profit'] = pnl
                df.loc[idx, 'home_goals'] = fixture_info.get('home_goals')
                df.loc[idx, 'away_goals'] = fixture_info.get('away_goals')

            updated += 1

        if not self.dry_run and updated > 0:
            df.to_csv(PAPER_TRADES_FILE, index=False)
            print(f"\n✓ Saved {updated} updates to paper_trades.csv")

        return {'updated': updated, 'errors': errors}

    def update_recommendations(self, filename: Optional[str] = None) -> Dict:
        """Update recommendation CSV files with results."""
        if filename:
            files = [RECOMMENDATIONS_DIR / filename]
        else:
            files = list(RECOMMENDATIONS_DIR.glob('rec_*.csv'))

        total_updated = 0
        total_errors = 0

        for filepath in files:
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)

            # Detect format: new format has 'status' column, old format has 'market' column
            is_new_format = 'status' in df.columns
            is_old_format = 'market' in df.columns and 'status' not in df.columns

            if is_new_format:
                pending = df[df['status'] == 'PENDING'].copy()
            elif is_old_format:
                # Old format - add status column if missing
                if 'result' not in df.columns:
                    df['result'] = None
                    df['actual'] = None
                pending = df[df['result'].isna()].copy()
            else:
                continue

            if len(pending) == 0:
                continue

            print(f"\nUpdating {filepath.name} ({len(pending)} pending)...")

            today = datetime.now().date()
            updated = 0

            for idx, row in pending.iterrows():
                # Parse date - handle both formats
                try:
                    if 'start_time' in row and pd.notna(row['start_time']):
                        match_date = pd.to_datetime(row['start_time']).date()
                    elif 'date' in row and pd.notna(row['date']):
                        match_date = pd.to_datetime(row['date']).date()
                    else:
                        continue
                except:
                    continue

                if match_date >= today:
                    continue

                # Need league info - try to infer from recent data
                league = self._infer_league(row['home_team'])
                if not league:
                    total_errors += 1
                    continue

                fixture_info = self._find_fixture(
                    row['home_team'],
                    row['away_team'],
                    str(match_date),
                    league
                )

                if not fixture_info:
                    total_errors += 1
                    continue

                fixture_id = fixture_info['fixture_id']

                # Check if match is finished
                if fixture_info.get('status') not in ['FT', 'AET', 'PEN']:
                    continue

                stats = self._get_match_stats(fixture_id, league)
                if stats is None:
                    stats = {'home': {}, 'away': {}}
                stats['home_goals'] = fixture_info.get('home_goals', 0)
                stats['away_goals'] = fixture_info.get('away_goals', 0)

                # Build bet_type string - handle both formats
                if is_new_format:
                    bet_type = f"{row['market']} {row['side']}"
                    line = row.get('line', 0) or 0
                else:
                    # Old format: market column + bet_type column (OVER/UNDER)
                    bet_type = f"{row['market']} {row.get('bet_type', '')}"
                    line = row.get('line', 0) or 0

                status, actual = self._evaluate_bet(bet_type, line, stats)
                odds = row.get('odds', 1.85)
                if pd.isna(odds):
                    odds = 1.85
                pnl = self._calculate_pnl(status, row.get('stake', 50) or 50, odds)

                print(f"  {row['home_team'][:12]:12} vs {row['away_team'][:12]:12} | "
                      f"{bet_type[:15]:15} | Actual: {actual:3} | {status}")

                if not self.dry_run:
                    if is_new_format:
                        df.loc[idx, 'status'] = status
                        df.loc[idx, 'actual_value'] = actual
                        df.loc[idx, 'pnl'] = pnl
                        df.loc[idx, 'settled_at'] = datetime.now().isoformat()
                    else:
                        df.loc[idx, 'result'] = status
                        df.loc[idx, 'actual'] = actual

                updated += 1
                total_updated += 1

            if not self.dry_run and updated > 0:
                df.to_csv(filepath, index=False)
                print(f"  ✓ Saved {updated} updates")

        return {'updated': total_updated, 'errors': total_errors}

    def _infer_league(self, team_name: str) -> Optional[str]:
        """Infer league from team name."""
        # Premier League teams
        pl_teams = ['manchester', 'liverpool', 'arsenal', 'chelsea', 'tottenham',
                    'newcastle', 'brighton', 'bournemouth', 'fulham', 'wolves',
                    'brentford', 'crystal', 'everton', 'west ham', 'burnley',
                    'nottingham', 'aston', 'leeds', 'sunderland']

        # La Liga teams
        la_liga_teams = ['barcelona', 'real madrid', 'atletico', 'sevilla', 'villarreal',
                        'real sociedad', 'athletic', 'valencia', 'betis', 'celta',
                        'getafe', 'osasuna', 'mallorca', 'alaves', 'girona',
                        'rayo', 'espanyol', 'levante', 'elche', 'oviedo']

        # Serie A teams
        serie_a_teams = ['juventus', 'inter', 'milan', 'napoli', 'roma', 'lazio',
                        'atalanta', 'fiorentina', 'torino', 'bologna', 'sassuolo',
                        'verona', 'udinese', 'lecce', 'genoa', 'cagliari',
                        'como', 'parma', 'cremonese', 'pisa']

        # Bundesliga teams
        bundesliga_teams = ['bayern', 'dortmund', 'leverkusen', 'leipzig', 'frankfurt',
                          'hoffenheim', 'wolfsburg', 'freiburg', 'union', 'stuttgart',
                          'gladbach', 'mainz', 'augsburg', 'werder', 'bochum', 'heidenheim']

        # Ligue 1 teams
        ligue_1_teams = ['psg', 'paris', 'marseille', 'lyon', 'monaco', 'lille',
                        'nice', 'lens', 'rennes', 'strasbourg', 'nantes',
                        'montpellier', 'toulouse', 'reims', 'le havre', 'auxerre']

        team_lower = team_name.lower()

        for t in pl_teams:
            if t in team_lower:
                return 'premier_league'
        for t in la_liga_teams:
            if t in team_lower:
                return 'la_liga'
        for t in serie_a_teams:
            if t in team_lower:
                return 'serie_a'
        for t in bundesliga_teams:
            if t in team_lower:
                return 'bundesliga'
        for t in ligue_1_teams:
            if t in team_lower:
                return 'ligue_1'

        return None

    def show_summary(self):
        """Show current P&L summary."""
        if not PAPER_TRADES_FILE.exists():
            return

        df = pd.read_csv(PAPER_TRADES_FILE)
        settled = df[df['status'].isin(['won', 'lost'])]

        if len(settled) == 0:
            print("\nNo settled bets yet.")
            return

        print("\n" + "=" * 60)
        print("P&L SUMMARY")
        print("=" * 60)

        total_stake = settled['stake'].sum()
        total_pnl = settled['profit'].sum()
        roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0
        wins = len(settled[settled['status'] == 'won'])
        losses = len(settled[settled['status'] == 'lost'])

        print(f"\nTotal bets: {len(settled)}")
        print(f"Won: {wins} | Lost: {losses} | Win rate: {wins/len(settled)*100:.1f}%")
        print(f"Total staked: ${total_stake:.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"ROI: {roi:+.1f}%")

        # By market
        if 'bet_type' in settled.columns:
            print("\n--- By Market ---")

            def get_market(bt):
                if 'Fouls' in str(bt):
                    return 'Fouls'
                elif 'Shots' in str(bt):
                    return 'Shots'
                elif 'Corner' in str(bt):
                    return 'Corners'
                elif 'Home' in str(bt) or 'Away' in str(bt):
                    return 'Match Result'
                return 'Other'

            settled['market'] = settled['bet_type'].apply(get_market)

            by_market = settled.groupby('market').agg({
                'stake': 'sum',
                'profit': 'sum',
                'status': lambda x: (x == 'won').sum()
            }).rename(columns={'status': 'wins'})
            by_market['bets'] = settled.groupby('market').size()
            by_market['roi'] = by_market['profit'] / by_market['stake'] * 100

            for market, row in by_market.iterrows():
                print(f"  {market:15} | {int(row['wins'])}/{int(row['bets'])} wins | "
                      f"P&L: ${row['profit']:+.2f} | ROI: {row['roi']:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Auto-update betting results')
    parser.add_argument('--dry-run', action='store_true', help='Preview without saving')
    parser.add_argument('--file', type=str, help='Specific recommendations file to update')
    parser.add_argument('--summary', action='store_true', help='Show P&L summary only')

    args = parser.parse_args()

    updater = ResultsUpdater(dry_run=args.dry_run)

    if args.summary:
        updater.show_summary()
        return

    if args.dry_run:
        print("DRY RUN - No changes will be saved")

    # Update paper trades
    result1 = updater.update_paper_trades()

    # Update recommendations
    result2 = updater.update_recommendations(args.file)

    print(f"\n{'=' * 60}")
    print(f"Total updated: {result1['updated'] + result2['updated']}")
    print(f"Errors: {result1['errors'] + result2['errors']}")

    # Show summary
    updater.show_summary()


if __name__ == '__main__':
    main()
