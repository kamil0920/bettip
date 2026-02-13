import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
import time
import os

from src.data_collection import FootballAPIClient

LEAGUES_CONFIG = {
    'premier_league': {
        'id': 39,
        'name': 'Premier League',
        'country': 'England',
        'folder': 'premier_league'
    },
    'la_liga': {
        'id': 140,
        'name': 'La Liga',
        'country': 'Spain',
        'folder': 'la_liga'
    },
    'bundesliga': {
        'id': 78,
        'name': 'Bundesliga',
        'country': 'Germany',
        'folder': 'bundesliga'
    },
    'serie_a': {
        'id': 135,
        'name': 'Serie A',
        'country': 'Italy',
        'folder': 'serie_a'
    },
    'ligue_1': {
        'id': 61,
        'name': 'Ligue 1',
        'country': 'France',
        'folder': 'ligue_1'
    },
    'ekstraklasa': {
        'id': 106,
        'name': 'Ekstraklasa',
        'country': 'Poland',
        'folder': 'ekstraklasa'
    },
    'mls': {
        'id': 253,
        'name': 'Major League Soccer',
        'country': 'USA',
        'folder': 'mls'
    },
    'liga_mx': {
        'id': 262,
        'name': 'Liga MX',
        'country': 'Mexico',
        'folder': 'liga_mx'
    },
    'eredivisie': {
        'id': 88,
        'name': 'Eredivisie',
        'country': 'Netherlands',
        'folder': 'eredivisie'
    },
    'portuguese_liga': {
        'id': 94,
        'name': 'Primeira Liga',
        'country': 'Portugal',
        'folder': 'portuguese_liga'
    },
    'turkish_super_lig': {
        'id': 203,
        'name': 'Süper Lig',
        'country': 'Turkey',
        'folder': 'turkish_super_lig'
    },
    'belgian_pro_league': {
        'id': 144,
        'name': 'Pro League',
        'country': 'Belgium',
        'folder': 'belgian_pro_league'
    },
    'scottish_premiership': {
        'id': 179,
        'name': 'Premiership',
        'country': 'Scotland',
        'folder': 'scottish_premiership'
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_RAW_DIR = PROJECT_ROOT / "data" / "01-raw"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixtures_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MatchDataCollector:
    """
    Collects match data using Parquet for storage.
    Operates on FLATTENED data structures (e.g. 'fixture.id' instead of 'fixture'['id']).
    """

    def __init__(self, base_data_dir: str = None):
        if base_data_dir:
            self.base_dir = Path(base_data_dir)
        else:
            self.base_dir = DEFAULT_DATA_RAW_DIR

        self.client = FootballAPIClient()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_fixtures_file_path(self, league_key: str, season: int) -> Path:
        """Get path to matches.parquet"""
        return self.get_season_dir(league_key, season) / 'matches.parquet'

    def get_season_dir(self, league_key: str, season: int) -> Path:
        league_config = LEAGUES_CONFIG[league_key]
        season_dir = self.base_dir / league_config['folder'] / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        return season_dir


    def _flatten_api_data(self, data: Any) -> Any:
        """
        Converts nested API JSON into flat dictionaries compatible with Parquet.
        Example: {'fixture': {'id': 1}} -> {'fixture.id': 1}
        """
        if not data:
            return data

        if isinstance(data, list):
            return pd.json_normalize(data).to_dict('records')

        if isinstance(data, dict):
            return pd.json_normalize([data]).iloc[0].to_dict()

        return data

    def _clean_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixes mixed types (int vs string) that crash Parquet."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric first — preserves NaN instead of creating 'None' strings
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().any() and converted.notna().sum() >= df[col].notna().sum() * 0.5:
                    df[col] = converted
                else:
                    df[col] = df[col].astype(str)
        return df

    def _append_to_parquet(self, filepath: Path, new_data: List[Dict]) -> None:
        """Appends list of flat dicts to a parquet file."""
        if not new_data: return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(new_data)

        if filepath.exists():
            try:
                existing_df = pd.read_parquet(filepath)
                # Drop all-NA columns before concat to avoid FutureWarning
                new_df = new_df.dropna(axis=1, how='all')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                self.logger.error(f"Read error {filepath}: {e}. Overwriting.")
                combined_df = new_df
        else:
            combined_df = new_df

        combined_df = self._clean_for_parquet(combined_df)
        combined_df.to_parquet(filepath, index=False)

    def _load_existing_ids(self, filepath: Path, id_column: str = 'fixture_id') -> Set[int]:
        """Fast check for existing IDs in a parquet file."""
        if not filepath.exists():
            return set()
        try:
            df = pd.read_parquet(filepath, columns=[id_column])
            return set(df[id_column].unique())
        except Exception:
            return set()

    def load_fixtures(self, league_key: str, season: int) -> Tuple[Dict, List[Dict]]:
        """Load flattened fixtures from Parquet."""
        file_path = self.get_fixtures_file_path(league_key, season)

        if not file_path.exists():
            return {}, []

        try:
            df = pd.read_parquet(file_path)
            return {'count': len(df)}, df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Error reading parquet: {e}")
            return {}, []

    def save_fixtures(self, league_key: str, season: int, fixtures: List[Dict]):
        """Save list of flat dicts to Parquet."""
        file_path = self.get_fixtures_file_path(league_key, season)

        try:
            df = pd.DataFrame(fixtures)
            df = self._clean_for_parquet(df)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(file_path, index=False)
            self.logger.info(f"✅ Saved {len(df)} matches to {file_path.name}")
            return True
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False

    def identify_fixtures_needing_update(self, fixtures: List[Dict], days_back: int = 30, include_future: int = 7) -> List[Dict]:
        """Finds matches to update using FLAT KEYS (fixture.date, etc.)"""
        to_update = []
        now = datetime.now()
        cutoff_past = now - timedelta(days=days_back)

        for f in fixtures:
            date_str = str(f.get('fixture.date', '')).replace('+00:00', '')
            try:
                f_date = datetime.fromisoformat(date_str)
            except ValueError:
                continue

            status = f.get('fixture.status.short')
            needs_update = False

            if status in ['NS', 'TBD', 'PST'] and f_date <= now:
                needs_update = True
            elif status in ['1H', '2H', 'HT', 'ET', 'P', 'LIVE', 'INT']:
                needs_update = True
            elif f_date >= cutoff_past and f_date <= now and status not in ['FT', 'AET', 'PEN']:
                needs_update = True
            elif f_date > now and (f_date - now).days <= include_future:
                last_upd = f.get('_last_api_update')
                if not last_upd:
                    needs_update = True
                else:
                    try:
                        last_date = datetime.fromisoformat(str(last_upd))
                        if (now - last_date).days >= 3:
                            needs_update = True
                    except:
                        needs_update = True

            if needs_update:
                to_update.append(f)

        self.logger.info(f"Found {len(to_update)} matches to update")
        return to_update

    def collect_fixture_details(self, fixture: Dict, league_key: str, season: int) -> Dict:
        """
        Collects events/lineups/players.
        Expects 'fixture' to be a FLAT dictionary.
        """
        fixture_id = fixture['fixture.id']
        home_team = fixture['teams.home.name']
        away_team = fixture['teams.away.name']
        season_dir = self.get_season_dir(league_key, season)

        base_info = {
            'fixture_id': fixture_id,
            'date': fixture['fixture.date'],
            'status': fixture['fixture.status.short'],
            'home_team': home_team,
            'away_team': away_team,
            'score_home': fixture.get('goals.home'),
            'score_away': fixture.get('goals.away')
        }

        stats = {'events': False, 'lineups': False, 'players': False, 'errors': []}

        try:
            events_file = season_dir / 'events.parquet'
            existing_ids = self._load_existing_ids(events_file)

            if fixture_id not in existing_ids:
                self.logger.info(f"  collecting events for {home_team} vs {away_team}")
                resp = self.client._make_request('/fixtures/events', {'fixture': fixture_id})
                data = resp.get('response', [])

                if data:
                    flat_rows = []
                    for item in data:
                        flat_item = self._flatten_api_data(item)
                        flat_rows.append({**base_info, **flat_item})

                    self._append_to_parquet(events_file, flat_rows)
                    stats['events'] = True
        except Exception as e:
            stats['errors'].append(f"events: {e}")

        try:
            lineups_file = season_dir / 'lineups.parquet'
            existing_ids = self._load_existing_ids(lineups_file)

            if fixture_id not in existing_ids:
                resp = self.client._make_request('/fixtures/lineups', {'fixture': fixture_id})
                data = resp.get('response', [])

                if data:
                    flat_rows = []
                    for team_data in data:
                        t_name = team_data['team']['name']
                        formation = team_data.get('formation')
                        coach = team_data.get('coach', {})
                        coach_name = coach.get('name') if isinstance(coach, dict) else None
                        coach_id = coach.get('id') if isinstance(coach, dict) else None
                        for p in team_data.get('startXI', []):
                            p_flat = self._flatten_api_data(p['player'])
                            flat_rows.append({**base_info, 'team_name': t_name, 'type': 'StartXI', 'formation': formation, 'coach_name': coach_name, 'coach_id': coach_id, **p_flat})
                        for p in team_data.get('substitutes', []):
                            p_flat = self._flatten_api_data(p['player'])
                            flat_rows.append({**base_info, 'team_name': t_name, 'type': 'Sub', 'formation': formation, 'coach_name': coach_name, 'coach_id': coach_id, **p_flat})

                    self._append_to_parquet(lineups_file, flat_rows)
                    stats['lineups'] = True
        except Exception as e:
            stats['errors'].append(f"lineups: {e}")

        try:
            stats_file = season_dir / 'player_stats.parquet'
            existing_ids = self._load_existing_ids(stats_file)

            if fixture_id not in existing_ids:
                resp = self.client._make_request('/fixtures/players', {'fixture': fixture_id})
                data = resp.get('response', [])

                if data:
                    flat_rows = []
                    for team_data in data:
                        t_name = team_data['team']['name']
                        for p_entry in team_data.get('players', []):
                            p_info = self._flatten_api_data(p_entry['player'])
                            for stat in p_entry['statistics']:
                                s_flat = self._flatten_api_data(stat)
                                flat_rows.append({**base_info, 'team_name': t_name, **p_info, **s_flat})

                    self._append_to_parquet(stats_file, flat_rows)
                    stats['players'] = True
        except Exception as e:
            stats['errors'].append(f"players: {e}")

        return stats

    def update_fixtures_smart(self, league_key: str, season: int, max_updates: int = None, days_back: int = 30):
        """Smart update using flat dicts."""
        _, fixtures = self.load_fixtures(league_key, season)

        # If no fixtures exist, fall back to full update
        if not fixtures:
            self.logger.info(f"No existing fixtures for {league_key} {season}, doing full update")
            return self.update_fixtures_full(league_key, season)

        fixtures_map = {f['fixture.id']: f for f in fixtures}

        to_update = self.identify_fixtures_needing_update(fixtures, days_back)

        if not to_update:
            self.logger.info("All up to date")
            return {'status': 'up_to_date'}

        if max_updates: to_update = to_update[:max_updates]

        stats = {'updated': 0, 'changed': 0, 'errors': 0}

        for f in to_update:
            fid = f['fixture.id']
            old_status = f.get('fixture.status.short', 'NS')
            old_score = (f.get('goals.home'), f.get('goals.away'))

            try:
                resp = self.client._make_request('/fixtures', {'id': fid})
                if resp and resp.get('response'):
                    nested_new = resp['response'][0]
                    nested_new['_last_api_update'] = datetime.now().isoformat()

                    flat_new = self._flatten_api_data(nested_new)

                    new_status = flat_new.get('fixture.status.short')
                    new_score = (flat_new.get('goals.home'), flat_new.get('goals.away'))

                    if new_status != old_status or new_score != old_score:
                        stats['changed'] += 1
                        if new_status in ['FT', 'AET', 'PEN']:
                            self.collect_fixture_details(flat_new, league_key, season)

                    fixtures_map[fid] = flat_new
                    stats['updated'] += 1
                    time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Update failed for {fid}: {e}")
                stats['errors'] += 1

        updated_list = list(fixtures_map.values())
        updated_list.sort(key=lambda x: x.get('fixture.date', ''))

        self.save_fixtures(league_key, season, updated_list)
        return stats

    def update_fixtures_full(self, league_key: str, season: int):
        """Redownloads all fixtures and saves as flat parquet."""
        self.logger.info(f"Full update {league_key} {season}")
        resp = self.client._make_request('/fixtures', {'league': LEAGUES_CONFIG[league_key]['id'], 'season': season})

        if resp and resp.get('response'):
            raw_list = resp['response']
            flat_list = self._flatten_api_data(raw_list)

            now_str = datetime.now().isoformat()
            for f in flat_list:
                f['_last_api_update'] = now_str

            self.save_fixtures(league_key, season, flat_list)
            return {'status': 'success', 'count': len(flat_list)}
        return {'status': 'error'}

    def update_live_fixtures(self, league_key: str, season: int) -> Dict:
        """Update only matches that are currently live or played today."""
        metadata, fixtures = self.load_fixtures(league_key, season)
        if not fixtures: return {'status': 'error'}

        today = datetime.now().date()
        to_update = []

        for f in fixtures:
            date_str = str(f.get('fixture.date', '')).replace('+00:00', '')
            f_date = datetime.fromisoformat(date_str).date()
            status = f.get('fixture.status.short')
            if f_date == today or status in ['LIVE', '1H', '2H', 'HT']:
                to_update.append(f)

        if not to_update:
            self.logger.info("No live matches found.")
            return {'status': 'no_matches'}

        return {'status': 'ok'}

    def analyze_fixtures_freshness(self, league_key: str, season: int) -> Dict:
        """
        analyzes the freshness of data in fixtures.parquet
        """
        metadata, fixtures = self.load_fixtures(league_key, season)

        if not fixtures:
            return {'error': 'No fixtures found'}

        now = datetime.now()
        analysis = {
            'total_fixtures': len(fixtures),
            'last_file_update': metadata.get('collected_at', 'Unknown'),
            'by_status': {},
            'needing_update': {
                'live_or_should_be': [],
                'recent_not_finished': [],
                'no_update_timestamp': [],
                'possibly_postponed': []
            },
            'statistics': {
                'finished': 0,
                'not_started': 0,
                'live': 0,
                'postponed': 0,
                'other': 0
            }
        }

        for fixture in fixtures:
            status = fixture.get('fixture.status.short')
            date_str = str(fixture.get('fixture.date', '')).replace('+00:00', '')
            try:
                fixture_date = datetime.fromisoformat(date_str)
            except:
                continue

            fid = fixture.get('fixture.id')
            home = fixture.get('teams.home.name')
            away = fixture.get('teams.away.name')

            if status not in analysis['by_status']:
                analysis['by_status'][status] = []
            analysis['by_status'][status].append({
                'id': fid,
                'date': fixture_date.strftime('%Y-%m-%d %H:%M'),
                'teams': f"{home} vs {away}"
            })

            if status in ['FT', 'AET', 'PEN']:
                analysis['statistics']['finished'] += 1
            elif status == 'NS':
                analysis['statistics']['not_started'] += 1
            elif status in ['1H', '2H', 'HT', 'ET', 'P', 'LIVE']:
                analysis['statistics']['live'] += 1
                analysis['needing_update']['live_or_should_be'].append(fid)
            elif status in ['PST', 'CANC', 'ABD', 'SUSP']:
                analysis['statistics']['postponed'] += 1
            else:
                analysis['statistics']['other'] += 1

            if status == 'NS' and fixture_date <= now:
                analysis['needing_update']['live_or_should_be'].append(fid)

            if (now - fixture_date).days <= 7 and status not in ['FT', 'AET', 'PEN']:
                analysis['needing_update']['recent_not_finished'].append(fid)

            if '_last_api_update' not in fixture:
                analysis['needing_update']['no_update_timestamp'].append(fid)

        total_needing_update = len(set(
            id for list_ids in analysis['needing_update'].values() for id in list_ids
        ))

        analysis['summary'] = {
            'total_needing_update': total_needing_update,
            'percentage_needing_update': (total_needing_update / len(fixtures)) * 100 if fixtures else 0
        }

        return analysis

    def _analyze_changes(self, old_fixtures_map: Dict, new_fixtures: List[Dict]) -> Dict:
        """
        analyze changes between updates
        """
        stats = {
            'total_fixtures': len(new_fixtures),
            'updated': len(new_fixtures),
            'changed': 0,
            'new_fixtures': 0,
            'changes': [],
            'status_changes': {},
            'api_calls': 1
        }

        for fixture in new_fixtures:
            fixture_id = fixture.get('fixture.id')

            if fixture_id not in old_fixtures_map:
                stats['new_fixtures'] += 1
                continue

            old = old_fixtures_map[fixture_id]

            old_status = old.get('fixture.status.short')
            new_status = fixture.get('fixture.status.short')

            if old_status != new_status:
                stats['changed'] += 1

                change_key = f"{old_status}→{new_status}"
                stats['status_changes'][change_key] = stats['status_changes'].get(change_key, 0) + 1

                home = fixture.get('teams.home.name')
                away = fixture.get('teams.away.name')

                stats['changes'].append({
                    'fixture_id': fixture_id,
                    'teams': f"{home} vs {away}",
                    'old_status': old_status,
                    'new_status': new_status,
                    'old_score': (old.get('goals.home'), old.get('goals.away')),
                    'new_score': (fixture.get('goals.home'), fixture.get('goals.away'))
                })

        return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='data collector'
    )

    parser.add_argument('--league', default='premier_league',
                        help='league key (default: premier_league)')
    parser.add_argument('--season', type=int, default=2025,
                        help='season (default: 2025)')
    parser.add_argument('--strategy', choices=['smart', 'full', 'live', 'analyze'],
                        default='smart', help='update strategy (default: smart)')
    parser.add_argument('--max-updates', type=int, default=None,
                        help='maximum number of matches to update (for smart)')
    parser.add_argument('--days-back', type=int, default=30,
                        help='how many days back to check (for smart)')
    parser.add_argument('--no-backup', action='store_true',
                        help='dont create a backup')

    args = parser.parse_args()

    if not os.getenv("DAILY_LIMIT"):
        os.environ["DAILY_LIMIT"] = "100"
    if not os.getenv("PER_MIN_LIMIT"):
        os.environ["PER_MIN_LIMIT"] = "10"

    updater = MatchDataCollector(str(DEFAULT_DATA_RAW_DIR))

    try:
        if args.strategy == 'analyze':
            logger.info("🔍 analyze freshness matches...")
            analysis = updater.analyze_fixtures_freshness(args.league, args.season)

            logger.info(f"\n📊 analyse matches.parquet:")
            logger.info(f"  • all matches: {analysis['total_fixtures']}")
            logger.info(f"  • last update file: {analysis['last_file_update']}")

            logger.info(f"\n📈 match stats:")
            for status_type, count in analysis['statistics'].items():
                logger.info(f"  • {status_type}: {count}")

            logger.info(f"\n⚠️ need update:")
            logger.info(f"  • live matches or should be: {len(analysis['needing_update']['live_or_should_be'])}")
            logger.info(f"  • recent unfinished: {len(analysis['needing_update']['recent_not_finished'])}")
            logger.info(f"  • no update timestamp: {len(analysis['needing_update']['no_update_timestamp'])}")

            logger.info(f"\n📊 summary:")
            logger.info(f"  • matches requiring updates: {analysis['summary']['total_needing_update']}")
            logger.info(f"  • percentage requiring update: {analysis['summary']['percentage_needing_update']:.1f}%")

        elif args.strategy == 'smart':
            updater.update_fixtures_smart(
                args.league,
                args.season,
                max_updates=args.max_updates,
                days_back=args.days_back
            )

        elif args.strategy == 'full':
            updater.update_fixtures_full(args.league, args.season)

        elif args.strategy == 'live':
            updater.update_live_fixtures(args.league, args.season)

        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠️ aborted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
