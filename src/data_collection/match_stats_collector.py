#!/usr/bin/env python
"""
Match Statistics Collector

Collects match-level statistics from API-Football including:
- Corner Kicks
- Shots (on goal, total, blocked)
- Ball Possession
- Fouls
- Offsides
- Passes

Usage:
    from src.data_collection.match_stats_collector import MatchStatsCollector

    collector = MatchStatsCollector()
    collector.collect_league_stats('premier_league', 2025)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

from src.data_collection.api_client import FootballAPIClient, APIError

logger = logging.getLogger(__name__)

# League ID mapping
LEAGUE_IDS = {
    'premier_league': 39,
    'la_liga': 140,
    'serie_a': 135,
    'bundesliga': 78,
    'ligue_1': 61,
}


class MatchStatsCollector:
    """Collects and stores match-level statistics."""

    def __init__(self, data_dir: str = "data/01-raw"):
        """
        Initialize the match stats collector.

        Args:
            data_dir: Base directory for raw data storage
        """
        self.data_dir = Path(data_dir)
        self.client = FootballAPIClient()

    def _parse_statistics(self, stats_response: List[Dict]) -> Dict[str, Any]:
        """
        Parse API response into structured statistics dict.

        Args:
            stats_response: Raw API response from /fixtures/statistics

        Returns:
            Dict with home/away statistics
        """
        result = {
            'home': {},
            'away': {},
        }

        if not stats_response or len(stats_response) < 2:
            return result

        for i, team_data in enumerate(stats_response[:2]):
            side = 'home' if i == 0 else 'away'
            team_info = team_data.get('team', {})
            statistics = team_data.get('statistics', [])

            result[side]['team_id'] = team_info.get('id')
            result[side]['team_name'] = team_info.get('name')

            # Parse each statistic
            for stat in statistics:
                stat_type = stat.get('type', '').lower().replace(' ', '_')
                value = stat.get('value')

                # Handle None values
                if value is None:
                    value = 0
                # Handle percentage values (e.g., "55%")
                elif isinstance(value, str) and '%' in value:
                    try:
                        value = int(value.replace('%', ''))
                    except ValueError:
                        value = 0
                # Handle string numbers (e.g., expected_goals "1.52")
                elif isinstance(value, str):
                    try:
                        # Try float first, then int if it's a whole number
                        float_val = float(value)
                        value = int(float_val) if float_val == int(float_val) else float_val
                    except ValueError:
                        value = 0

                result[side][stat_type] = value

        return result

    def collect_fixture_stats(self, fixture_id: int) -> Optional[Dict]:
        """
        Collect statistics for a single fixture.

        Args:
            fixture_id: The fixture ID to collect stats for

        Returns:
            Parsed statistics dict or None if failed
        """
        try:
            response = self.client.get_fixture_statistics(fixture_id)
            if response:
                stats = self._parse_statistics(response)
                stats['fixture_id'] = fixture_id
                stats['collected_at'] = datetime.now().isoformat()
                return stats
            return None
        except APIError as e:
            logger.warning(f"Failed to get stats for fixture {fixture_id}: {e}")
            return None

    def collect_league_stats(
        self,
        league: str,
        season: int,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Collect match statistics for all completed fixtures in a league/season.

        Args:
            league: League name (e.g., 'premier_league')
            season: Season year (e.g., 2025)
            force_refresh: If True, re-fetch all stats even if file exists

        Returns:
            DataFrame with match statistics
        """
        league_dir = self.data_dir / league / str(season)
        stats_path = league_dir / "match_stats.parquet"

        # Load existing stats if available
        existing_stats = {}
        if stats_path.exists() and not force_refresh:
            try:
                existing_df = pd.read_parquet(stats_path)
                existing_stats = {
                    row['fixture_id']: row.to_dict()
                    for _, row in existing_df.iterrows()
                }
                logger.info(f"Loaded {len(existing_stats)} existing stats")
            except Exception as e:
                logger.warning(f"Failed to load existing stats: {e}")

        # Load matches to get fixture IDs
        matches_path = league_dir / "matches.parquet"
        if not matches_path.exists():
            raise FileNotFoundError(f"Matches file not found: {matches_path}")

        matches = pd.read_parquet(matches_path)

        # Filter to completed matches only
        completed = matches[matches['fixture.status.short'] == 'FT']
        logger.info(f"Found {len(completed)} completed matches")

        # Collect stats for each fixture
        all_stats = []
        new_count = 0

        for _, match in completed.iterrows():
            fixture_id = match['fixture.id']

            # Skip if already have stats
            if fixture_id in existing_stats:
                all_stats.append(existing_stats[fixture_id])
                continue

            # Fetch new stats
            stats = self.collect_fixture_stats(fixture_id)
            if stats:
                # Add match context
                stats['date'] = match['fixture.date']
                stats['home_team'] = match['teams.home.name']
                stats['away_team'] = match['teams.away.name']
                stats['home_goals'] = match['goals.home']
                stats['away_goals'] = match['goals.away']

                all_stats.append(stats)
                new_count += 1

                if new_count % 10 == 0:
                    logger.info(f"Collected {new_count} new fixture stats...")

        # Convert to DataFrame
        if all_stats:
            # Flatten nested dicts
            flat_stats = []
            for s in all_stats:
                flat = {
                    'fixture_id': s.get('fixture_id'),
                    'date': s.get('date'),
                    'home_team': s.get('home_team'),
                    'away_team': s.get('away_team'),
                    'home_goals': s.get('home_goals'),
                    'away_goals': s.get('away_goals'),
                    'collected_at': s.get('collected_at'),
                }

                # Add home stats with prefix
                for key, value in s.get('home', {}).items():
                    flat[f'home_{key}'] = value

                # Add away stats with prefix
                for key, value in s.get('away', {}).items():
                    flat[f'away_{key}'] = value

                flat_stats.append(flat)

            df = pd.DataFrame(flat_stats)

            # Ensure consistent types for numeric columns to avoid parquet errors
            for col in df.columns:
                if col in ['fixture_id', 'home_goals', 'away_goals']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif col not in ['date', 'home_team', 'away_team', 'collected_at',
                                 'home_team_name', 'away_team_name']:
                    # Convert all other columns to numeric (int or float)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Save to parquet
            league_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(stats_path, index=False)
            logger.info(f"Saved {len(df)} match stats to {stats_path}")

            return df

        return pd.DataFrame()

    def get_corner_summary(self, league: str, season: int) -> Dict:
        """
        Get summary statistics for corners in a league/season.

        Args:
            league: League name
            season: Season year

        Returns:
            Dict with corner statistics summary
        """
        stats_path = self.data_dir / league / str(season) / "match_stats.parquet"

        if not stats_path.exists():
            return {'error': 'No match stats file found'}

        df = pd.read_parquet(stats_path)

        # Check if corner data exists
        if 'home_corner_kicks' not in df.columns:
            return {'error': 'No corner data in stats file'}

        # Calculate totals
        df['total_corners'] = df['home_corner_kicks'] + df['away_corner_kicks']

        summary = {
            'matches': len(df),
            'avg_total_corners': df['total_corners'].mean(),
            'std_total_corners': df['total_corners'].std(),
            'avg_home_corners': df['home_corner_kicks'].mean(),
            'avg_away_corners': df['away_corner_kicks'].mean(),
            'over_9_5_rate': (df['total_corners'] > 9.5).mean(),
            'over_10_5_rate': (df['total_corners'] > 10.5).mean(),
            'over_11_5_rate': (df['total_corners'] > 11.5).mean(),
        }

        return summary


def main():
    """Run match stats collection."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Collect match statistics')
    parser.add_argument('--league', default='premier_league', help='League name')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--force', action='store_true', help='Force refresh all')

    args = parser.parse_args()

    collector = MatchStatsCollector()

    print(f"\n{'='*70}")
    print(f"MATCH STATISTICS COLLECTOR")
    print(f"{'='*70}")
    print(f"\nLeague: {args.league}")
    print(f"Season: {args.season}")

    df = collector.collect_league_stats(args.league, args.season, args.force)

    if len(df) > 0:
        print(f"\nCollected stats for {len(df)} matches")

        # Show corner summary if available
        if 'home_corner_kicks' in df.columns:
            print(f"\n{'='*70}")
            print("CORNER STATISTICS SUMMARY")
            print(f"{'='*70}")

            summary = collector.get_corner_summary(args.league, args.season)
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("\nNo corner data found in collected stats")
    else:
        print("\nNo stats collected")


if __name__ == "__main__":
    main()
