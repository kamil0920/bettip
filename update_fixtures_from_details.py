#!/usr/bin/env python3
"""
DEPRECATED: This script is for legacy JSON format.
With the migration to Parquet storage, use match_collector.py methods instead.

Update fixtures.json based on detailed data files (events, lineups, players)
"""
import warnings
warnings.warn(
    "update_fixtures_from_details.py is deprecated. "
    "Data is now stored in Parquet format. Use match_collector.py methods instead.",
    DeprecationWarning
)
import json
from pathlib import Path
from datetime import datetime
import re

def extract_fixture_id(filename):
    """Extract fixture ID from filename like 'fixture_1379063_events.json'"""
    match = re.search(r'fixture_(\d+)_', filename)
    return int(match.group(1)) if match else None

def load_fixture_details(fixture_id, season_dir):
    """Load details from events/lineups/players files"""
    details = {}

    # Try to load from players file (has most complete info)
    players_file = season_dir / 'players' / f'fixture_{fixture_id}_players.json'
    if players_file.exists():
        with open(players_file, 'r') as f:
            file_data = json.load(f)
            # Check if fixture_info is in data wrapper
            data = file_data.get('data', file_data)
            if 'fixture_info' in data:
                details['score'] = data['fixture_info'].get('score', {})
                details['status'] = data['fixture_info'].get('status', 'FT')
                return details

    # Try lineups
    lineups_file = season_dir / 'lineups' / f'fixture_{fixture_id}_lineups.json'
    if lineups_file.exists():
        with open(lineups_file, 'r') as f:
            file_data = json.load(f)
            data = file_data.get('data', file_data)
            if 'fixture_info' in data:
                details['score'] = data['fixture_info'].get('score', {})
                details['status'] = data['fixture_info'].get('status', 'FT')
                return details

    # Try events
    events_file = season_dir / 'events' / f'fixture_{fixture_id}_events.json'
    if events_file.exists():
        with open(events_file, 'r') as f:
            file_data = json.load(f)
            data = file_data.get('data', file_data)
            if 'fixture_info' in data:
                details['score'] = data['fixture_info'].get('score', {})
                details['status'] = 'FT'  # If events exist, match is finished
                return details

    return details

def update_fixtures_from_details(season_dir):
    """Update fixtures.json based on detailed data files"""

    fixtures_file = season_dir / 'fixtures.json'

    if not fixtures_file.exists():
        print(f"❌ fixtures.json not found in {season_dir}")
        return False

    # Load fixtures.json
    print(f"Loading {fixtures_file}...")
    with open(fixtures_file, 'r') as f:
        fixtures_data = json.load(f)

    metadata = fixtures_data.get('metadata', {})
    fixtures = fixtures_data.get('data', [])

    print(f"Found {len(fixtures)} fixtures in fixtures.json")

    # Find all fixture IDs that have detailed data
    fixture_ids_with_details = set()

    for folder in ['events', 'lineups', 'players']:
        folder_path = season_dir / folder
        if folder_path.exists():
            for file in folder_path.glob('fixture_*_*.json'):
                fixture_id = extract_fixture_id(file.name)
                if fixture_id:
                    fixture_ids_with_details.add(fixture_id)

    print(f"Found {len(fixture_ids_with_details)} fixtures with detailed data")

    # Update fixtures
    updated_count = 0
    changes = []

    for fixture in fixtures:
        fixture_id = fixture['fixture']['id']

        if fixture_id in fixture_ids_with_details:
            old_status = fixture['fixture']['status']['short']
            old_score = (fixture['goals'].get('home'), fixture['goals'].get('away'))
            old_winner = (fixture['teams']['home'].get('winner'), fixture['teams']['away'].get('winner'))

            # Load details
            details = load_fixture_details(fixture_id, season_dir)

            if details:
                new_status = details.get('status', 'FT')
                new_score = details.get('score', {})

                # Update fixture
                fixture['fixture']['status']['short'] = new_status
                if new_status == 'FT':
                    fixture['fixture']['status']['long'] = 'Match Finished'

                if new_score:
                    home_goals = new_score.get('home')
                    away_goals = new_score.get('away')

                    fixture['goals']['home'] = home_goals
                    fixture['goals']['away'] = away_goals

                    # Update winner field
                    if home_goals is not None and away_goals is not None:
                        if home_goals > away_goals:
                            fixture['teams']['home']['winner'] = True
                            fixture['teams']['away']['winner'] = False
                        elif away_goals > home_goals:
                            fixture['teams']['home']['winner'] = False
                            fixture['teams']['away']['winner'] = True
                        else:  # Draw
                            fixture['teams']['home']['winner'] = None
                            fixture['teams']['away']['winner'] = None

                # Mark as updated
                fixture['_last_api_update'] = datetime.now().isoformat()

                # Check if changed
                new_score_tuple = (new_score.get('home'), new_score.get('away'))
                new_winner = (fixture['teams']['home'].get('winner'), fixture['teams']['away'].get('winner'))

                if old_status != new_status or old_score != new_score_tuple or old_winner != new_winner:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']
                    changes.append({
                        'fixture_id': fixture_id,
                        'teams': f"{home_team} vs {away_team}",
                        'old_status': old_status,
                        'new_status': new_status,
                        'old_score': old_score,
                        'new_score': new_score_tuple,
                        'winner': new_winner
                    })
                    updated_count += 1

    # Update metadata
    metadata['last_update'] = datetime.now().isoformat()
    metadata['updated_from_details'] = True

    # Create backup
    backup_file = fixtures_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    print(f"Creating backup: {backup_file.name}")
    with open(backup_file, 'w') as f:
        json.dump(fixtures_data, f, indent=2, ensure_ascii=False)

    # Save updated fixtures.json
    updated_data = {
        'metadata': metadata,
        'data': fixtures
    }

    with open(fixtures_file, 'w') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Updated {updated_count} fixtures")

    if changes:
        print(f"\n📊 Changes made:")
        for change in changes[:10]:  # Show first 10
            winner_str = ""
            if change['winner'][0] is True:
                winner_str = " [HOME WIN]"
            elif change['winner'][1] is True:
                winner_str = " [AWAY WIN]"
            elif change['winner'][0] is None:
                winner_str = " [DRAW]"

            print(f"  • {change['teams']}: {change['old_status']}→{change['new_status']}, "
                  f"score: {change['old_score']}→{change['new_score']}{winner_str}")

        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")

    return True

if __name__ == "__main__":
    season_dir = Path("data/01-raw/premier_league/2025")

    if not season_dir.exists():
        print(f"❌ Directory not found: {season_dir}")
        exit(1)

    success = update_fixtures_from_details(season_dir)
    exit(0 if success else 1)