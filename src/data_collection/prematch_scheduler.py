"""
Pre-match data collection scheduler.

Monitors upcoming matches and triggers data collection at optimal times:
- Injuries: Daily refresh + pre-match update
- Lineups: ~1 hour before kickoff (when available)
- Weather: ~2 hours before kickoff
- Predictions: Can be collected anytime, refresh pre-match

Usage:
    # Run as daemon (checks every 15 minutes)
    python -m src.data_collection.prematch_scheduler --daemon

    # One-time check for matches in next N hours
    python -m src.data_collection.prematch_scheduler --hours 6

    # Process specific fixture
    python -m src.data_collection.prematch_scheduler --fixture 1234567
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_collection.prematch_collector import PreMatchCollector, LEAGUE_IDS

logger = logging.getLogger(__name__)


# Collection windows (hours before kickoff)
COLLECTION_WINDOWS = {
    'early': {  # 24-6 hours before
        'injuries': True,
        'predictions': True,
        'h2h': True,
        'lineups': False,  # Not available yet
        'weather': False,  # Too early
    },
    'pre_match': {  # 6-2 hours before
        'injuries': True,  # Refresh
        'predictions': True,
        'h2h': False,  # Already collected
        'lineups': False,  # Usually not available
        'weather': True,  # Weather forecast
    },
    'lineup_window': {  # 1-0.5 hours before
        'injuries': True,  # Final check
        'predictions': False,
        'h2h': False,
        'lineups': True,  # NOW AVAILABLE
        'weather': True,  # Final weather
    },
}


class PreMatchScheduler:
    """
    Scheduler for pre-match data collection.

    Monitors upcoming fixtures and collects data at optimal times.
    """

    def __init__(
        self,
        output_dir: str = 'data/06-prematch',
        state_file: str = 'data/06-prematch/scheduler_state.json',
    ):
        self.collector = PreMatchCollector()
        self.output_dir = Path(output_dir)
        self.state_file = Path(state_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load scheduler state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return {
            'last_check': None,
            'collected_fixtures': {},  # fixture_id -> {phase: timestamp}
        }

    def _save_state(self) -> None:
        """Save scheduler state to file."""
        try:
            self.state['last_check'] = datetime.now(timezone.utc).isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def get_upcoming_matches(
        self,
        hours_ahead: int = 24,
        leagues: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get upcoming matches within the specified time window.

        Args:
            hours_ahead: Hours to look ahead
            leagues: List of league keys to check (default: all)

        Returns:
            DataFrame with upcoming fixtures
        """
        leagues = leagues or list(LEAGUE_IDS.keys())
        all_fixtures = []

        for league in leagues:
            league_id = LEAGUE_IDS.get(league)
            if not league_id:
                continue

            try:
                # Get current season
                season = datetime.now().year
                fixtures = self.collector.get_upcoming_fixtures(league_id, season, next_n=20)

                if not fixtures.empty:
                    fixtures['league'] = league
                    all_fixtures.append(fixtures)

            except Exception as e:
                logger.error(f"Failed to get fixtures for {league}: {e}")

        if not all_fixtures:
            return pd.DataFrame()

        combined = pd.concat(all_fixtures, ignore_index=True)

        # Filter by time window
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        # Ensure datetime column is timezone-aware
        if combined['date'].dt.tz is None:
            combined['date'] = combined['date'].dt.tz_localize('UTC')

        filtered = combined[
            (combined['date'] >= now) &
            (combined['date'] <= cutoff)
        ].sort_values('date')

        return filtered

    def determine_collection_phase(self, match_time: datetime) -> Optional[str]:
        """
        Determine which collection phase applies based on time to kickoff.

        Args:
            match_time: Match kickoff time (timezone-aware)

        Returns:
            Phase name ('early', 'pre_match', 'lineup_window') or None
        """
        now = datetime.now(timezone.utc)

        if match_time.tzinfo is None:
            match_time = match_time.replace(tzinfo=timezone.utc)

        hours_to_kickoff = (match_time - now).total_seconds() / 3600

        if hours_to_kickoff > 24:
            return None  # Too far out
        elif hours_to_kickoff > 6:
            return 'early'
        elif hours_to_kickoff > 1:
            return 'pre_match'
        elif hours_to_kickoff > 0.25:  # 15 mins before
            return 'lineup_window'
        else:
            return None  # Match started

    def should_collect(
        self,
        fixture_id: int,
        phase: str,
        min_interval_hours: float = 2.0
    ) -> bool:
        """
        Check if we should collect data for this fixture/phase.

        Args:
            fixture_id: Fixture ID
            phase: Collection phase
            min_interval_hours: Minimum hours between collections for same phase

        Returns:
            True if collection should proceed
        """
        collected = self.state.get('collected_fixtures', {})
        fixture_state = collected.get(str(fixture_id), {})

        if phase not in fixture_state:
            return True

        last_collected = datetime.fromisoformat(fixture_state[phase])
        hours_since = (datetime.now(timezone.utc) - last_collected).total_seconds() / 3600

        # For lineup_window, always collect (lineups appear suddenly)
        if phase == 'lineup_window':
            return hours_since > 0.25  # At least 15 mins

        return hours_since > min_interval_hours

    def collect_for_fixture(
        self,
        fixture_id: int,
        phase: str,
        fixture_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Collect pre-match data for a fixture based on phase.

        Args:
            fixture_id: Fixture ID
            phase: Collection phase
            fixture_info: Optional fixture metadata

        Returns:
            Collected data dict
        """
        config = COLLECTION_WINDOWS.get(phase, {})
        result = {
            'fixture_id': fixture_id,
            'phase': phase,
            'collected_at': datetime.now(timezone.utc).isoformat(),
            'fixture_info': fixture_info,
        }

        logger.info(f"Collecting {phase} data for fixture {fixture_id}")

        # Injuries
        if config.get('injuries'):
            try:
                injuries = self.collector.get_injuries_by_fixture(fixture_id)
                result['injuries'] = injuries.to_dict('records') if not injuries.empty else []
                result['injuries_count'] = len(injuries)
                logger.info(f"  Injuries: {len(injuries)}")
            except Exception as e:
                logger.error(f"  Injuries failed: {e}")
                result['injuries'] = []

        # Predictions
        if config.get('predictions'):
            try:
                predictions = self.collector.get_predictions(fixture_id)
                result['predictions'] = predictions
                logger.info(f"  Predictions: {'Yes' if predictions else 'No'}")
            except Exception as e:
                logger.error(f"  Predictions failed: {e}")
                result['predictions'] = {}

        # H2H
        if config.get('h2h') and fixture_info:
            try:
                home_id = fixture_info.get('home_team_id')
                away_id = fixture_info.get('away_team_id')
                if home_id and away_id:
                    h2h_summary = self.collector.get_h2h_summary(home_id, away_id)
                    result['h2h_summary'] = h2h_summary
                    logger.info(f"  H2H: {h2h_summary.get('matches', 0)} matches")
            except Exception as e:
                logger.error(f"  H2H failed: {e}")
                result['h2h_summary'] = {}

        # Lineups
        if config.get('lineups'):
            try:
                lineups = self.collector.get_lineups(fixture_id)
                result['lineups'] = lineups
                result['lineups_available'] = lineups.get('available', False)
                logger.info(f"  Lineups: {'Available' if lineups.get('available') else 'Not yet'}")
            except Exception as e:
                logger.error(f"  Lineups failed: {e}")
                result['lineups'] = {'available': False}

        # Save to file
        self._save_fixture_data(fixture_id, phase, result)

        # Update state
        if str(fixture_id) not in self.state['collected_fixtures']:
            self.state['collected_fixtures'][str(fixture_id)] = {}
        self.state['collected_fixtures'][str(fixture_id)][phase] = result['collected_at']
        self._save_state()

        return result

    def _save_fixture_data(
        self,
        fixture_id: int,
        phase: str,
        data: Dict[str, Any]
    ) -> None:
        """Save fixture data to file."""
        # Create fixture directory
        fixture_dir = self.output_dir / str(fixture_id)
        fixture_dir.mkdir(parents=True, exist_ok=True)

        # Save phase data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{phase}_{timestamp}.json"

        with open(fixture_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        # Also save as "latest" for easy access
        with open(fixture_dir / f"{phase}_latest.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def run_check(
        self,
        hours_ahead: int = 24,
        leagues: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a single check for upcoming matches.

        Args:
            hours_ahead: Hours to look ahead
            leagues: List of leagues to check

        Returns:
            List of collection results
        """
        logger.info(f"Checking for matches in next {hours_ahead} hours")

        fixtures = self.get_upcoming_matches(hours_ahead, leagues)

        if fixtures.empty:
            logger.info("No upcoming matches found")
            return []

        logger.info(f"Found {len(fixtures)} upcoming matches")

        results = []
        for _, fixture in fixtures.iterrows():
            fixture_id = fixture['fixture_id']
            match_time = fixture['date']
            home = fixture['home_team_name']
            away = fixture['away_team_name']

            phase = self.determine_collection_phase(match_time)

            if not phase:
                continue

            if not self.should_collect(fixture_id, phase):
                logger.debug(f"Skipping {home} vs {away} - already collected for {phase}")
                continue

            logger.info(f"\n{home} vs {away} ({match_time}) - Phase: {phase}")

            try:
                result = self.collect_for_fixture(
                    fixture_id,
                    phase,
                    fixture.to_dict()
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to collect for {fixture_id}: {e}")

        return results

    def run_daemon(
        self,
        check_interval_minutes: int = 15,
        hours_ahead: int = 24,
        leagues: Optional[List[str]] = None
    ) -> None:
        """
        Run as daemon, checking periodically.

        Args:
            check_interval_minutes: Minutes between checks
            hours_ahead: Hours to look ahead
            leagues: List of leagues to monitor
        """
        logger.info(f"Starting daemon (interval: {check_interval_minutes}min)")

        while True:
            try:
                self.run_check(hours_ahead, leagues)
            except Exception as e:
                logger.error(f"Check failed: {e}")

            logger.info(f"Sleeping for {check_interval_minutes} minutes...")
            time.sleep(check_interval_minutes * 60)


def generate_prematch_recommendations(
    fixture_id: int,
    output_dir: str = 'data/06-prematch'
) -> Optional[Dict[str, Any]]:
    """
    Generate betting recommendations using latest pre-match data.

    Args:
        fixture_id: Fixture ID
        output_dir: Directory with pre-match data

    Returns:
        Recommendation dict or None
    """
    fixture_dir = Path(output_dir) / str(fixture_id)

    if not fixture_dir.exists():
        logger.warning(f"No pre-match data for fixture {fixture_id}")
        return None

    # Load latest data from each phase
    data = {}
    for phase in ['early', 'pre_match', 'lineup_window']:
        phase_file = fixture_dir / f"{phase}_latest.json"
        if phase_file.exists():
            with open(phase_file, 'r') as f:
                data[phase] = json.load(f)

    if not data:
        return None

    # Use most recent data
    latest = data.get('lineup_window') or data.get('pre_match') or data.get('early')

    recommendations = {
        'fixture_id': fixture_id,
        'fixture_info': latest.get('fixture_info', {}),
        'data_phase': list(data.keys())[-1],
        'signals': [],
    }

    # Analyze injuries
    injuries = latest.get('injuries', [])
    home_injuries = sum(1 for i in injuries if i.get('team_id') == latest.get('fixture_info', {}).get('home_team_id'))
    away_injuries = len(injuries) - home_injuries

    if home_injuries > away_injuries + 2:
        recommendations['signals'].append({
            'market': 'Away +0.25 AH',
            'confidence': 0.55,
            'reason': f'Home team has {home_injuries} injuries vs {away_injuries}',
        })
    elif away_injuries > home_injuries + 2:
        recommendations['signals'].append({
            'market': 'Home -0.25 AH',
            'confidence': 0.55,
            'reason': f'Away team has {away_injuries} injuries vs {home_injuries}',
        })

    # Analyze predictions
    predictions = latest.get('predictions', {})
    if predictions:
        percent = predictions.get('percent', {})

        # Strong away signal
        away_pct = float(percent.get('away', '0%').replace('%', '')) / 100
        if away_pct >= 0.50:
            recommendations['signals'].append({
                'market': 'Away Win',
                'confidence': away_pct,
                'reason': f"API predicts {away_pct*100:.0f}% away win",
            })

        # Strong home signal
        home_pct = float(percent.get('home', '0%').replace('%', '')) / 100
        if home_pct >= 0.60:
            recommendations['signals'].append({
                'market': 'Home Win',
                'confidence': home_pct,
                'reason': f"API predicts {home_pct*100:.0f}% home win",
            })

    # Analyze lineups
    lineups = latest.get('lineups', {})
    if lineups.get('available'):
        home_formation = lineups.get('home', {}).get('formation', '')
        away_formation = lineups.get('away', {}).get('formation', '')

        # Attacking formations suggest goals
        attacking = ['4-3-3', '3-4-3', '4-2-4']
        if any(f in home_formation for f in attacking) and any(f in away_formation for f in attacking):
            recommendations['signals'].append({
                'market': 'Over 2.5 Goals',
                'confidence': 0.55,
                'reason': f'Both teams attacking formations: {home_formation} vs {away_formation}',
            })

    return recommendations


# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-match data scheduler')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--hours', type=int, default=24, help='Hours ahead to check')
    parser.add_argument('--interval', type=int, default=15, help='Check interval (minutes)')
    parser.add_argument('--fixture', type=int, help='Specific fixture ID')
    parser.add_argument('--phase', type=str, default='lineup_window',
                       choices=['early', 'pre_match', 'lineup_window'])
    parser.add_argument('--leagues', type=str, nargs='+',
                       default=['premier_league'],
                       help='Leagues to monitor')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    scheduler = PreMatchScheduler()

    if args.fixture:
        result = scheduler.collect_for_fixture(args.fixture, args.phase)
        print(json.dumps(result, indent=2, default=str))
    elif args.daemon:
        scheduler.run_daemon(args.interval, args.hours, args.leagues)
    else:
        results = scheduler.run_check(args.hours, args.leagues)
        print(f"\nCollected data for {len(results)} fixtures")
