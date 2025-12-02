import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
import time
import os
from enum import Enum

from src.data_collection.api_client import FootballAPIClient
from src.data_collection.collector import LEAGUES_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "01-raw"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixtures_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    FULL = "full"
    SMART = "smart"
    RECENT = "recent"
    LIVE = "live"
    ROUND = "round"


@dataclass
class FixtureUpdateInfo:
    fixture_id: int
    old_status: str
    new_status: str
    old_score: Optional[Tuple[int, int]]
    new_score: Optional[Tuple[int, int]]
    round: str
    date: str

    @property
    def has_changed(self) -> bool:
        """check if change has occurred"""
        return (self.old_status != self.new_status or
                self.old_score != self.new_score)

    @property
    def status_changed(self) -> bool:
        """check if status has changed"""
        return self.old_status != self.new_status

    @property
    def score_changed(self) -> bool:
        """check if result has changed"""
        return self.old_score != self.new_score


class MatchDataCollector:
    """
    minimizes api usage by selectively update
    """

    def __init__(self, base_data_dir: str = "data/01-raw"):
        self.base_dir = Path(base_data_dir)
        self.client = FootballAPIClient()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._matches_cache = {}

    def save_json(self, data: Any, filepath: Path, pretty: bool = True) -> None:
        """
        save data to JSON file with metadata

        Args:
            data: data to save
            filepath: output file path
            pretty: wWhether to format JSON with indentation
        """
        output_data = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'records_count': len(data) if isinstance(data, list) else 1,
                'api_usage': self.client.state.get('count', 0),
                'daily_limit': self.client.daily_limit
            },
            'data': data
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(output_data, f, ensure_ascii=False, default=str)

        self.logger.info(f"saved to: {filepath}")

    def get_fixtures_file_path(self, league_key: str, season: int) -> Path:
        """get fixtures.json path"""
        return self.base_dir / LEAGUES_CONFIG[league_key]['folder'] / str(season) / 'fixtures.json'

    def get_season_dir(self, league_key: str, season: int) -> Path:
        """get directory for specific league and season"""
        league_config = LEAGUES_CONFIG[league_key]
        return self.base_dir / league_config['folder'] / str(season)

    def collect_fixture_details(self, fixture: Dict, league_key: str, season: int) -> Dict:
        """
        collect detailed data (events, lineups, players) for a single fixture
        saves each to separate files

        Args:
            fixture: fixture data
            league_key: league key
            season: season year

        Returns:
            Dict with collection stats
        """
        fixture_id = fixture['fixture']['id']
        season_dir = self.get_season_dir(league_key, season)

        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        fixture_date = fixture['fixture']['date']

        stats = {
            'events': False,
            'lineups': False,
            'players': False,
            'errors': []
        }

        try:
            events_dir = season_dir / 'events'
            events_dir.mkdir(exist_ok=True)
            events_file = events_dir / f'fixture_{fixture_id}_events.json'

            if not events_file.exists():
                self.logger.info(f"  collecting events for {home_team} vs {away_team}...")
                events_response = self.client._make_request('/fixtures/events', {'fixture': fixture_id})
                events_data = events_response.get('response', [])

                if events_data:
                    events_package = {
                        'fixture_info': {
                            'id': fixture_id,
                            'date': fixture_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'score': fixture.get('goals', {})
                        },
                        'events': events_data
                    }
                    self.save_json(events_package, events_file)
                    stats['events'] = True
                    self.logger.info(f"  ✓ saved {len(events_data)} events")
                else:
                    self.logger.debug(f"  no events data for fixture {fixture_id}")
            else:
                self.logger.debug(f"  events already exist for fixture {fixture_id}")
                stats['events'] = True

        except Exception as e:
            self.logger.warning(f"  failed to collect events: {e}")
            stats['errors'].append(f"events: {e}")

        try:
            lineups_dir = season_dir / 'lineups'
            lineups_dir.mkdir(exist_ok=True)
            lineup_file = lineups_dir / f'fixture_{fixture_id}_lineups.json'

            if not lineup_file.exists():
                self.logger.info(f"  collecting lineups for {home_team} vs {away_team}...")
                lineups_response = self.client._make_request('/fixtures/lineups', {'fixture': fixture_id})
                lineup_data = lineups_response.get('response', [])

                if lineup_data and len(lineup_data) > 0:
                    total_players = sum(
                        len(team_lineup.get('startXI', [])) + len(team_lineup.get('substitutes', []))
                        for team_lineup in lineup_data
                    )

                    lineup_package = {
                        'fixture_info': {
                            'id': fixture_id,
                            'date': fixture_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'score': fixture.get('goals', {}),
                            'status': fixture['fixture']['status']['short']
                        },
                        'lineups': lineup_data
                    }
                    self.save_json(lineup_package, lineup_file)
                    stats['lineups'] = True
                    self.logger.info(f"  ✓ saved lineups ({total_players} players)")
                else:
                    self.logger.debug(f"  no lineup data for fixture {fixture_id}")
            else:
                self.logger.debug(f"  lineups already exist for fixture {fixture_id}")
                stats['lineups'] = True

        except Exception as e:
            self.logger.warning(f"  failed to collect lineups: {e}")
            stats['errors'].append(f"lineups: {e}")

        try:
            players_dir = season_dir / 'players'
            players_dir.mkdir(exist_ok=True)
            players_file = players_dir / f'fixture_{fixture_id}_players.json'

            if not players_file.exists():
                self.logger.info(f"  collecting player statistics for {home_team} vs {away_team}...")
                players_response = self.client._make_request('/fixtures/players', {'fixture': fixture_id})
                players_data = players_response.get('response', [])

                if players_data and len(players_data) > 0:
                    total_players = sum(
                        len(team.get('players', [])) for team in players_data
                    )

                    players_package = {
                        'fixture_info': {
                            'id': fixture_id,
                            'date': fixture_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'score': fixture.get('goals', {}),
                            'status': fixture['fixture']['status']['short']
                        },
                        'players': players_data
                    }
                    self.save_json(players_package, players_file)
                    stats['players'] = True
                    self.logger.info(f"  ✓ saved player stats ({total_players} players)")
                else:
                    self.logger.debug(f"  no player statistics for fixture {fixture_id}")
            else:
                self.logger.debug(f"  player stats already exist for fixture {fixture_id}")
                stats['players'] = True

        except Exception as e:
            self.logger.warning(f"  failed to collect player statistics: {e}")
            stats['errors'].append(f"players: {e}")

        return stats

    def load_fixtures(self, league_key: str, season: int) -> Tuple[Dict, List[Dict]]:
        """
        load fixtures.json with metadata

        Returns:
            Tuple (metadata, fixtures_list)
        """
        file_path = self.get_fixtures_file_path(league_key, season)

        if not file_path.exists():
            self.logger.error(f"no fixtures.json file for {league_key}/{season}")
            return {}, []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                fixtures = data.get('data', [])

                self.logger.info(f"loaded {len(fixtures)} matches from {file_path.name}")
                self.logger.info(f"  last update: {metadata.get('collected_at', 'unknown')}")

                return metadata, fixtures
        except json.JSONDecodeError as e:
            self.logger.error(f"parse error fixtures.json: {e}")
            return {}, []

    def save_fixtures(self, league_key: str, season: int, fixtures: List[Dict], metadata: Optional[Dict] = None, backup: bool = True) -> bool:
        """
        save updated matches to file

        Args:
            league_key: league key
            season: season
            fixtures: list of matches to save
            metadata: metada (optional, will be updated)
            backup: back up old file?

        Returns:
            True if saved
        """
        file_path = self.get_fixtures_file_path(league_key, season)

        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"created backup: {backup_path.name}")
            except Exception as e:
                self.logger.warning(f"backup failed: {e}")

        if metadata is None:
            metadata = {}

        metadata.update({
            'collected_at': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'records_count': len(fixtures),
            'api_usage': self.client.state.get('count', 0),
            'daily_limit': self.client.daily_limit
        })

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                'metadata': metadata,
                'data': fixtures
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"✅ saved {len(fixtures)} matches to {file_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"save error fixtures.json: {e}")
            return False

    def identify_fixtures_needing_update(self, fixtures: List[Dict], days_back: int = 30, include_live: bool = True, include_future_next_days: int = 7) -> List[Dict]:
        """
        finds matches requiring update

        Args:
            fixtures: list of all fixtures
            days_back: number of days back to check
            include_live: whether to include live matches
            include_future_next_days: how many days in future to check (for date changes)

        Returns:
            list of fixtures requiring updating
        """
        fixtures_to_update = []
        now = datetime.now()
        cutoff_past = now - timedelta(days=days_back)
        cutoff_future = now + timedelta(days=include_future_next_days)

        for fixture in fixtures:
            fixture_date_str = fixture['fixture']['date'].replace('+00:00', '')
            fixture_date = datetime.fromisoformat(fixture_date_str)
            status = fixture['fixture']['status']['short']

            needs_update = False
            reason = ""

            if status in ['NS', 'TBD', 'PST'] and fixture_date <= now:
                needs_update = True
                reason = "match should have started by now"

            elif status in ['1H', '2H', 'HT', 'ET', 'P', 'LIVE', 'INT']:
                needs_update = True
                reason = "match in progress"

            elif fixture_date >= cutoff_past and fixture_date <= now and status not in ['FT', 'AET', 'PEN']:
                needs_update = True
                reason = f"match from last {days_back} days unfinished"

            elif fixture_date > now and fixture_date <= cutoff_future and include_future_next_days > 0:
                last_update = fixture.get('_last_api_update')
                if last_update:
                    last_update_date = datetime.fromisoformat(last_update)
                    if (now - last_update_date).days >= 3:
                        needs_update = True
                        reason = "check if match has been postponed"
                else:
                    needs_update = True
                    reason = "no information about last update"

            if needs_update:
                fixtures_to_update.append(fixture)
                self.logger.debug(f"to be updated: {fixture['teams']['home']['name']} vs "
                                  f"{fixture['teams']['away']['name']} ({fixture_date.strftime('%Y-%m-%d')}) "
                                  f"- {reason}")

        self.logger.info(f"found {len(fixtures_to_update)} matches requiring updates")
        return fixtures_to_update

    def update_fixtures_smart(self, league_key: str, season: int, max_updates: Optional[int] = None, days_back: int = 30) -> Dict:
        """
        smart update
        it uses many api queries but only for specific matches

        Args:
            league_key: league key
            season: season
            max_updates: maximum number of matches to update
            days_back: how many days back should I check

        Returns:
            update stats
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(" smart match updates ".center(70, "="))
        self.logger.info("=" * 70)

        metadata, fixtures = self.load_fixtures(league_key, season)

        if not fixtures:
            self.logger.error("no fixtures to update")
            return {'status': 'error', 'message': 'No fixtures found'}

        fixtures_map = {f['fixture']['id']: f for f in fixtures}

        fixtures_to_update = self.identify_fixtures_needing_update(fixtures, days_back)

        if not fixtures_to_update:
            self.logger.info("✅ all fixtures are up to date")
            return {'status': 'up_to_date', 'checked': len(fixtures)}

        if max_updates and len(fixtures_to_update) > max_updates:
            fixtures_to_update = fixtures_to_update[:max_updates]
            self.logger.info(f"limited to {max_updates} update")

        remaining = self.client.daily_limit - self.client.state.get('count', 0)
        if remaining < len(fixtures_to_update):
            self.logger.warning(f"insufficient api limit. remain: {remaining}, "f"need: {len(fixtures_to_update)}")
            fixtures_to_update = fixtures_to_update[:remaining]

        stats = {
            'total_fixtures': len(fixtures),
            'checked': len(fixtures_to_update),
            'updated': 0,
            'changed': 0,
            'errors': 0,
            'changes': []
        }

        self.logger.info(f"update {len(fixtures_to_update)} matches...")

        details_stats = {
            'events_collected': 0,
            'lineups_collected': 0,
            'players_collected': 0,
            'details_errors': 0
        }

        for idx, fixture in enumerate(fixtures_to_update, 1):
            fixture_id = fixture['fixture']['id']
            old_status = fixture['fixture']['status']['short']
            old_score = (fixture['goals']['home'], fixture['goals']['away'])

            self.logger.info(f"[{idx}/{len(fixtures_to_update)}] update match {fixture_id}")

            try:
                response = self.client._make_request(
                    '/fixtures',
                    {'id': fixture_id}
                )

                if response and response.get('response'):
                    updated_fixture = response['response'][0]

                    updated_fixture['_last_api_update'] = datetime.now().isoformat()
                    updated_fixture = self._sanitize_fixture_data(updated_fixture)

                    new_status = updated_fixture['fixture']['status']['short']
                    new_score = (updated_fixture['goals']['home'], updated_fixture['goals']['away'])

                    update_info = FixtureUpdateInfo(
                        fixture_id=fixture_id,
                        old_status=old_status,
                        new_status=new_status,
                        old_score=old_score if old_score != (None, None) else None,
                        new_score=new_score if new_score != (None, None) else None,
                        round=fixture['league']['round'],
                        date=fixture['fixture']['date']
                    )

                    if update_info.has_changed:
                        self.logger.info(f"  ✓ change: {old_status}→{new_status}, "
                                         f"score: {old_score}→{new_score}")
                        stats['changed'] += 1
                        stats['changes'].append(asdict(update_info))
                    else:
                        self.logger.debug(f"  - no changes (status: {new_status})")

                    fixtures_map[fixture_id] = updated_fixture
                    stats['updated'] += 1

                    if new_status in ['FT', 'AET', 'PEN']:
                        fixture_date = datetime.fromisoformat(updated_fixture['fixture']['date'].replace('+00:00', ''))
                        if fixture_date <= datetime.now():
                            self.logger.info(f"  collecting detailed data for finished match...")
                            detail_stats = self.collect_fixture_details(updated_fixture, league_key, season)

                            if detail_stats['events']:
                                details_stats['events_collected'] += 1
                            if detail_stats['lineups']:
                                details_stats['lineups_collected'] += 1
                            if detail_stats['players']:
                                details_stats['players_collected'] += 1
                            if detail_stats['errors']:
                                details_stats['details_errors'] += len(detail_stats['errors'])

                            time.sleep(7)

                else:
                    self.logger.warning(f"  ⚠ no data for match {fixture_id}")
                    stats['errors'] += 1

                if idx < len(fixtures_to_update):
                    time.sleep(7)

            except Exception as e:
                self.logger.error(f"  ✗ no update match {fixture_id}: {e}")
                stats['errors'] += 1

        stats.update(details_stats)

        updated_fixtures_list = list(fixtures_map.values())
        updated_fixtures_list.sort(key=lambda x: x['fixture']['date'])

        if self.save_fixtures(league_key, season, updated_fixtures_list, metadata):
            self.logger.info("✅ matches updated and saved")
        else:
            self.logger.error("❌ error save match")
            stats['save_error'] = True

        self._log_update_summary(stats)

        return stats

    def update_fixtures_full(self, league_key: str, season: int) -> Dict:
        """
        full update

        Returns:
            update stats
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(" full update ".center(70, "="))
        self.logger.info("=" * 70)
        self.logger.info(f"league: {LEAGUES_CONFIG[league_key]['name']}")
        self.logger.info(f"season: {season}/{season + 1}")

        old_metadata, old_fixtures = self.load_fixtures(league_key, season)
        old_fixtures_map = {f['fixture']['id']: f for f in old_fixtures} if old_fixtures else {}

        try:
            self.logger.info("download all matches from the api...")

            response = self.client._make_request(
                '/fixtures',
                {
                    'league': LEAGUES_CONFIG[league_key]['id'],
                    'season': season
                }
            )

            if not response or not response.get('response'):
                self.logger.error("no api response")
                return {'status': 'error', 'message': 'No API response'}

            new_fixtures = response['response']
            self.logger.info(f"downloaded {len(new_fixtures)} matches")

            clean_fixtures = []
            for fixture in new_fixtures:
                clean = self._sanitize_fixture_data(fixture)
                clean['_last_api_update'] = datetime.now().isoformat()
                clean_fixtures.append(clean)

            new_fixtures = clean_fixtures

            stats = self._analyze_changes(old_fixtures_map, new_fixtures)

            if self.save_fixtures(league_key, season, new_fixtures):
                self.logger.info("✅ fixtures.json updated and saved")
                stats['status'] = 'success'
            else:
                self.logger.error("❌ save error fixtures.json")
                stats['status'] = 'error'
                stats['message'] = 'Save failed'

            self._log_update_summary(stats)

            return stats

        except Exception as e:
            self.logger.error(f"update failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def update_live_fixtures(self, league_key: str, season: int) -> Dict:
        """
        live strategy

        Returns:
            update stats
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info(" update live matches ".center(70, "="))
        self.logger.info("=" * 70)

        metadata, fixtures = self.load_fixtures(league_key, season)

        if not fixtures:
            return {'status': 'error', 'message': 'No fixtures found'}

        today = datetime.now().date()
        fixtures_to_update = []

        for fixture in fixtures:
            fixture_date = datetime.fromisoformat(
                fixture['fixture']['date'].replace('+00:00', '')
            ).date()
            status = fixture['fixture']['status']['short']

            if fixture_date == today or status in ['1H', '2H', 'HT', 'ET', 'P', 'LIVE']:
                fixtures_to_update.append(fixture)

        if not fixtures_to_update:
            self.logger.info("no matches today or live")
            return {'status': 'no_matches', 'checked': 0}

        self.logger.info(f"found {len(fixtures_to_update)} matches to update")

        return self.update_fixtures_smart(
            league_key,
            season,
            max_updates=len(fixtures_to_update),
            days_back=1
        )

    def _sanitize_fixture_data(self, match_data: Dict) -> Dict:
        """
        Remove detailed data (events, lineups, players, statistics) from match object
        to keep fixtures.json lightweight and consistent.
        """
        allowed_keys = {'fixture', 'league', 'teams', 'goals', 'score', '_last_api_update'}

        clean_data = {k: v for k, v in match_data.items() if k in allowed_keys}

        if '_last_api_update' not in clean_data and '_last_api_update' in match_data:
            clean_data['_last_api_update'] = match_data['_last_api_update']

        return clean_data

    def _analyze_changes(self, old_fixtures_map: Dict, new_fixtures: List[Dict]) -> Dict:
        """
        analyze changes between changes

        Returns:
            dict with change stats
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
            fixture_id = fixture['fixture']['id']

            if fixture_id not in old_fixtures_map:
                stats['new_fixtures'] += 1
                continue

            old = old_fixtures_map[fixture_id]

            old_status = old['fixture']['status']['short']
            new_status = fixture['fixture']['status']['short']

            if old_status != new_status:
                stats['changed'] += 1

                change_key = f"{old_status}→{new_status}"
                stats['status_changes'][change_key] = stats['status_changes'].get(change_key, 0) + 1

                stats['changes'].append({
                    'fixture_id': fixture_id,
                    'teams': f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}",
                    'old_status': old_status,
                    'new_status': new_status,
                    'old_score': (old['goals']['home'], old['goals']['away']),
                    'new_score': (fixture['goals']['home'], fixture['goals']['away'])
                })

        return stats

    def _log_update_summary(self, stats: Dict):
        self.logger.info("\n" + "=" * 70)
        self.logger.info(" update summary ".center(70, "="))
        self.logger.info("=" * 70)

        self.logger.info(f"📊 stats:")
        self.logger.info(f"  • all matches: {stats.get('total_fixtures', 'N/A')}")
        self.logger.info(f"  • chacked: {stats.get('checked', stats.get('updated', 0))}")
        self.logger.info(f"  • updated: {stats.get('updated', 0)}")
        self.logger.info(f"  • changed: {stats.get('changed', 0)}")

        if stats.get('new_fixtures'):
            self.logger.info(f"  • new fixtures: {stats['new_fixtures']}")

        if stats.get('errors'):
            self.logger.warning(f"  • errors: {stats['errors']}")

        if stats.get('api_calls'):
            self.logger.info(f"  • api queries used: {stats['api_calls']}")

        if stats.get('events_collected') or stats.get('lineups_collected') or stats.get('players_collected'):
            self.logger.info("\n📦 detailed data collected:")
            if stats.get('events_collected'):
                self.logger.info(f"  • events: {stats['events_collected']} matches")
            if stats.get('lineups_collected'):
                self.logger.info(f"  • lineups: {stats['lineups_collected']} matches")
            if stats.get('players_collected'):
                self.logger.info(f"  • player stats: {stats['players_collected']} matches")
            if stats.get('details_errors'):
                self.logger.warning(f"  • detail errors: {stats['details_errors']}")

        if stats.get('status_changes'):
            self.logger.info("\n📈 status changes:")
            for change, count in sorted(stats['status_changes'].items(),
                                        key=lambda x: x[1], reverse=True):
                self.logger.info(f"  • {change}: {count} matches")

        if stats.get('changes'):
            self.logger.info(f"\n🔄 example changes (max 5):")
            for change in stats['changes'][:5]:
                if isinstance(change, dict) and 'teams' in change:
                    self.logger.info(f"  • {change['teams']}: "
                                     f"{change['old_status']}→{change['new_status']}, "
                                     f"score: {change['old_score']}→{change['new_score']}")

    def analyze_fixtures_freshness(self, league_key: str, season: int) -> Dict:
        """
        analyzes the freshness of data in fixtures.json
        shows which matches may require updating

        Returns:
            data freshness analysis
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
            status = fixture['fixture']['status']['short']
            fixture_date = datetime.fromisoformat(fixture['fixture']['date'].replace('+00:00', ''))

            if status not in analysis['by_status']:
                analysis['by_status'][status] = []
            analysis['by_status'][status].append({
                'id': fixture['fixture']['id'],
                'date': fixture_date.strftime('%Y-%m-%d %H:%M'),
                'teams': f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}"
            })

            if status in ['FT', 'AET', 'PEN']:
                analysis['statistics']['finished'] += 1
            elif status == 'NS':
                analysis['statistics']['not_started'] += 1
            elif status in ['1H', '2H', 'HT', 'ET', 'P', 'LIVE']:
                analysis['statistics']['live'] += 1
                analysis['needing_update']['live_or_should_be'].append(fixture['fixture']['id'])
            elif status in ['PST', 'CANC', 'ABD', 'SUSP']:
                analysis['statistics']['postponed'] += 1
            else:
                analysis['statistics']['other'] += 1

            if status == 'NS' and fixture_date <= now:
                analysis['needing_update']['live_or_should_be'].append(fixture['fixture']['id'])

            if (now - fixture_date).days <= 7 and status not in ['FT', 'AET', 'PEN']:
                analysis['needing_update']['recent_not_finished'].append(fixture['fixture']['id'])

            if '_last_api_update' not in fixture:
                analysis['needing_update']['no_update_timestamp'].append(fixture['fixture']['id'])

        total_needing_update = len(set(
            id for list_ids in analysis['needing_update'].values() for id in list_ids
        ))

        analysis['summary'] = {
            'total_needing_update': total_needing_update,
            'percentage_needing_update': (total_needing_update / len(fixtures)) * 100 if fixtures else 0
        }

        return analysis


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

    updater = MatchDataCollector(str(DATA_RAW_DIR))

    try:
        if args.strategy == 'analyze':
            logger.info("🔍 analyze freshness matches...")
            analysis = updater.analyze_fixtures_freshness(args.league, args.season)

            logger.info(f"\n📊 analyse fixtures.json:")
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
            stats = updater.update_fixtures_smart(
                args.league,
                args.season,
                max_updates=args.max_updates,
                days_back=args.days_back
            )

        elif args.strategy == 'full':
            stats = updater.update_fixtures_full(args.league, args.season)

        elif args.strategy == 'live':
            stats = updater.update_live_fixtures(args.league, args.season)

        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠️ aborted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())