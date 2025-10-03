#!/usr/bin/env python3
"""
Football Data Audit - Check what data was collected and identify gaps
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FootballDataAuditor:
    """Audit collected football data and generate reports."""

    def __init__(self, base_data_dir: str = "football_data"):
        self.base_dir = Path(base_data_dir)

    def load_json(self, filepath: Path) -> Optional[Dict]:
        """Load JSON file safely."""
        if not filepath.exists():
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def audit_league_season(self, league_folder: str, season: int) -> Dict:
        """Audit data for a specific league and season."""
        season_dir = self.base_dir / league_folder / str(season)

        audit_result = {
            'league': league_folder,
            'season': season,
            'season_dir': str(season_dir),
            'exists': season_dir.exists(),
            'basic_files': {},
            'lineups': {'collected': 0, 'failed': 0, 'total_fixtures': 0},
            'events': {'collected': 0, 'failed': 0, 'total_fixtures': 0},
            'total_size_mb': 0,
            'collection_info': {},
            'errors': []
        }

        if not season_dir.exists():
            audit_result['errors'].append("Season directory doesn't exist")
            return audit_result

        # Check basic files
        basic_files = ['teams.json', 'fixtures.json', 'standings.json', 'collection_summary.json']

        for file_name in basic_files:
            file_path = season_dir / file_name
            if file_path.exists():
                file_size = file_path.stat().st_size
                data = self.load_json(file_path)

                record_count = 0
                if data and 'data' in data:
                    if isinstance(data['data'], list):
                        record_count = len(data['data'])
                    else:
                        record_count = 1

                audit_result['basic_files'][file_name] = {
                    'exists': True,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / 1024 / 1024, 2),
                    'records': record_count,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                audit_result['total_size_mb'] += file_size / 1024 / 1024
            else:
                audit_result['basic_files'][file_name] = {'exists': False}

        # Get collection info from summary
        summary_path = season_dir / 'collection_summary.json'
        if summary_path.exists():
            summary_data = self.load_json(summary_path)
            if summary_data and 'data' in summary_data:
                audit_result['collection_info'] = summary_data['data']

        # Audit lineups
        lineups_dir = season_dir / 'lineups'
        if lineups_dir.exists():
            lineup_files = list(lineups_dir.glob('fixture_*_lineups.json'))
            audit_result['lineups']['collected'] = len(lineup_files)

            # Calculate total size of lineups
            lineup_size = sum(f.stat().st_size for f in lineup_files)
            audit_result['lineups']['size_mb'] = round(lineup_size / 1024 / 1024, 2)
            audit_result['total_size_mb'] += lineup_size / 1024 / 1024

        # Audit events
        events_dir = season_dir / 'events'
        if events_dir.exists():
            event_files = list(events_dir.glob('fixture_*_events.json'))
            audit_result['events']['collected'] = len(event_files)

            # Calculate total size of events
            events_size = sum(f.stat().st_size for f in event_files)
            audit_result['events']['size_mb'] = round(events_size / 1024 / 1024, 2)
            audit_result['total_size_mb'] += events_size / 1024 / 1024

        # Get total fixtures from fixtures.json to calculate completion rates
        fixtures_data = self.load_json(season_dir / 'fixtures.json')
        if fixtures_data and 'data' in fixtures_data:
            fixtures = fixtures_data['data']
            completed_fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

            audit_result['lineups']['total_fixtures'] = len(completed_fixtures)
            audit_result['events']['total_fixtures'] = len(completed_fixtures)

            if len(completed_fixtures) > 0:
                audit_result['lineups']['completion_rate'] = round(
                    audit_result['lineups']['collected'] / len(completed_fixtures) * 100, 1
                )
                audit_result['events']['completion_rate'] = round(
                    audit_result['events']['collected'] / len(completed_fixtures) * 100, 1
                )

        audit_result['total_size_mb'] = round(audit_result['total_size_mb'], 2)
        return audit_result

    def audit_all_data(self) -> Dict:
        """Audit all collected data."""
        print("🔍 FOOTBALL DATA AUDIT")
        print("=" * 50)

        if not self.base_dir.exists():
            print(f"❌ Data directory doesn't exist: {self.base_dir}")
            return {}

        # Find all leagues and seasons
        audit_results = {}
        total_size = 0
        total_fixtures_with_lineups = 0
        total_fixtures_with_events = 0

        for league_dir in self.base_dir.iterdir():
            if league_dir.is_dir():
                league_name = league_dir.name
                audit_results[league_name] = {}

                print(f"\n🏆 {league_name.upper()}")
                print("-" * 30)

                for season_dir in sorted(league_dir.iterdir()):
                    if season_dir.is_dir() and season_dir.name.isdigit():
                        season = int(season_dir.name)

                        audit_result = self.audit_league_season(league_name, season)
                        audit_results[league_name][season] = audit_result

                        # Print season summary
                        self.print_season_summary(audit_result)

                        total_size += audit_result['total_size_mb']
                        total_fixtures_with_lineups += audit_result['lineups']['collected']
                        total_fixtures_with_events += audit_result['events']['collected']

        # Print overall summary
        print(f"\n📊 OVERALL SUMMARY")
        print("=" * 50)
        print(f"📁 Total data size: {total_size:.2f} MB")
        print(f"👥 Total fixtures with lineups: {total_fixtures_with_lineups}")
        print(f"⚽ Total fixtures with events: {total_fixtures_with_events}")

        # Save detailed audit report
        self.save_audit_report(audit_results)

        return audit_results

    def print_season_summary(self, audit_result: Dict):
        """Print summary for a single season."""
        season = audit_result['season']

        if not audit_result['exists']:
            print(f"  {season}: ❌ No data")
            return

        basic_files_count = sum(1 for f in audit_result['basic_files'].values() if f.get('exists', False))

        lineups_info = audit_result['lineups']
        events_info = audit_result['events']

        print(f"  {season}: ✅ Basic files: {basic_files_count}/4 | "
              f"Lineups: {lineups_info['collected']}/{lineups_info['total_fixtures']} "
              f"({lineups_info.get('completion_rate', 0)}%) | "
              f"Events: {events_info['collected']}/{events_info['total_fixtures']} "
              f"({events_info.get('completion_rate', 0)}%) | "
              f"Size: {audit_result['total_size_mb']} MB")

    def save_audit_report(self, audit_results: Dict):
        """Save detailed audit report to JSON and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full JSON report
        json_report_path = self.base_dir / f'audit_report_{timestamp}.json'
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(audit_results, f, indent=2, default=str)

        print(f"💾 Detailed report saved: {json_report_path}")

        # Create CSV summary
        csv_data = []
        for league_name, seasons in audit_results.items():
            for season, data in seasons.items():
                csv_data.append({
                    'league': league_name,
                    'season': season,
                    'basic_files': sum(1 for f in data['basic_files'].values() if f.get('exists', False)),
                    'fixtures_total': data.get('basic_files', {}).get('fixtures.json', {}).get('records', 0),
                    'lineups_collected': data['lineups']['collected'],
                    'lineups_completion_rate': data['lineups'].get('completion_rate', 0),
                    'events_collected': data['events']['collected'],
                    'events_completion_rate': data['events'].get('completion_rate', 0),
                    'total_size_mb': data['total_size_mb'],
                    'collection_date': data.get('collection_info', {}).get('collection_date', 'Unknown')
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_report_path = self.base_dir / f'audit_summary_{timestamp}.csv'
            df.to_csv(csv_report_path, index=False)
            print(f"📊 CSV summary saved: {csv_report_path}")

    def find_missing_data(self) -> Dict:
        """Find missing or incomplete data."""
        print(f"\n🔍 FINDING MISSING DATA")
        print("=" * 30)

        missing_data = {
            'seasons_without_basic_files': [],
            'seasons_with_low_lineup_completion': [],
            'seasons_with_no_events': [],
            'large_size_seasons': [],
            'recent_collections': []
        }

        for league_dir in self.base_dir.iterdir():
            if league_dir.is_dir():
                league_name = league_dir.name

                for season_dir in sorted(league_dir.iterdir()):
                    if season_dir.is_dir() and season_dir.name.isdigit():
                        season = int(season_dir.name)
                        audit_result = self.audit_league_season(league_name, season)

                        # Check for missing basic files
                        basic_files_count = sum(
                            1 for f in audit_result['basic_files'].values() if f.get('exists', False))
                        if basic_files_count < 4:
                            missing_data['seasons_without_basic_files'].append({
                                'league': league_name,
                                'season': season,
                                'missing_files': basic_files_count
                            })

                        # Check lineup completion rate
                        lineup_rate = audit_result['lineups'].get('completion_rate', 0)
                        if 0 < lineup_rate < 50:  # Has some but less than 50%
                            missing_data['seasons_with_low_lineup_completion'].append({
                                'league': league_name,
                                'season': season,
                                'completion_rate': lineup_rate
                            })

                        # Check for seasons with no events
                        if audit_result['events']['collected'] == 0 and audit_result['lineups']['total_fixtures'] > 0:
                            missing_data['seasons_with_no_events'].append({
                                'league': league_name,
                                'season': season,
                                'completed_fixtures': audit_result['lineups']['total_fixtures']
                            })

                        # Check for unusually large data
                        if audit_result['total_size_mb'] > 100:
                            missing_data['large_size_seasons'].append({
                                'league': league_name,
                                'season': season,
                                'size_mb': audit_result['total_size_mb']
                            })

        # Print findings
        for category, items in missing_data.items():
            if items:
                print(f"\n⚠️  {category.replace('_', ' ').title()}:")
                for item in items:
                    print(f"   {item}")

        return missing_data


def check_data_quality():
    """Check the quality of specific data files."""
    print(f"\n🔍 DATA QUALITY CHECK")
    print("=" * 30)

    auditor = FootballDataAuditor()

    # Sample some files to check data quality
    sample_checks = []

    for league_dir in auditor.base_dir.iterdir():
        if league_dir.is_dir():
            league_name = league_dir.name

            # Check most recent season
            season_dirs = [d for d in league_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if season_dirs:
                latest_season_dir = sorted(season_dirs, key=lambda x: int(x.name))[-1]
                season = int(latest_season_dir.name)

                # Check fixtures file
                fixtures_file = latest_season_dir / 'fixtures.json'
                if fixtures_file.exists():
                    fixtures_data = auditor.load_json(fixtures_file)
                    if fixtures_data and 'data' in fixtures_data:
                        fixtures = fixtures_data['data']

                        # Analyze fixture statuses
                        status_counts = {}
                        for fixture in fixtures:
                            status = fixture['fixture']['status']['short']
                            status_counts[status] = status_counts.get(status, 0) + 1

                        sample_checks.append({
                            'league': league_name,
                            'season': season,
                            'total_fixtures': len(fixtures),
                            'fixture_statuses': status_counts,
                            'file_size_mb': round(fixtures_file.stat().st_size / 1024 / 1024, 2)
                        })

                # Check a sample lineup file
                lineups_dir = latest_season_dir / 'lineups'
                if lineups_dir.exists():
                    lineup_files = list(lineups_dir.glob('fixture_*_lineups.json'))
                    if lineup_files:
                        sample_lineup = auditor.load_json(lineup_files[0])
                        if sample_lineup and 'data' in sample_lineup:
                            lineups = sample_lineup['data']['lineups']

                            total_players = 0
                            for team_lineup in lineups:
                                if 'startXI' in team_lineup:
                                    total_players += len(team_lineup['startXI'])
                                if 'substitutes' in team_lineup:
                                    total_players += len(team_lineup['substitutes'])

                            print(f"📋 Sample lineup check for {league_name} {season}:")
                            print(f"   File: {lineup_files[0].name}")
                            print(f"   Teams: {len(lineups)}")
                            print(f"   Total players: {total_players}")

    # Print sample checks
    for check in sample_checks:
        print(f"\n📊 {check['league'].upper()} {check['season']}:")
        print(f"   Fixtures: {check['total_fixtures']} ({check['file_size_mb']} MB)")
        print(f"   Statuses: {check['fixture_statuses']}")


def main():
    """Main audit function."""
    auditor = FootballDataAuditor()

    # Full audit
    audit_results = auditor.audit_all_data()

    # Find missing data
    missing_data = auditor.find_missing_data()

    # Check data quality
    check_data_quality()

    print(f"\n✅ Audit completed!")
    print(f"📁 Check {auditor.base_dir} for detailed reports")


if __name__ == "__main__":
    main()
