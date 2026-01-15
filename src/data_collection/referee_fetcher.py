#!/usr/bin/env python
"""
Referee Assignment Fetcher

Fetches referee assignments for upcoming Premier League matches from external sources.

Sources:
- referee-equipment.com blog posts
- premierleague.com/referees

Usage:
    from src.data_collection.referee_fetcher import RefereeFetcher

    fetcher = RefereeFetcher()
    assignments = fetcher.fetch_current_matchweek()
    fetcher.update_fixtures_with_referees(assignments)
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class RefereeFetcher:
    """Fetches and updates referee assignments for fixtures."""

    REFEREE_EQUIPMENT_BASE = "https://referee-equipment.com/blogs/the-ah-blog"
    PL_REFEREES_URL = "https://www.premierleague.com/referees/overview"

    # Team name normalization mapping (scraped name -> dataset name)
    TEAM_ALIASES = {
        'AFC Bournemouth': 'Bournemouth',
        'Brighton & Hove Albion': 'Brighton',
        'Brighton & Hove Albion FC': 'Brighton',
        'Leeds United AFC': 'Leeds',
        'Leeds United': 'Leeds',
        'Newcastle United': 'Newcastle',
        'Newcastle United FC': 'Newcastle',
        'Nottingham Forest': 'Nottingham Forest',
        'Nottm Forest': 'Nottingham Forest',
        'Tottenham Hotspur': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves',
        'Wolverhampton Wanderers FC': 'Wolves',
        'West Ham United': 'West Ham',
        'Sunderland AFC': 'Sunderland',
    }

    # Referee surname -> full name mapping (Premier League referees)
    REFEREE_NAMES = {
        'Taylor': 'Anthony Taylor',
        'Oliver': 'Michael Oliver',
        'Kavanagh': 'Chris Kavanagh',
        'Pawson': 'Craig Pawson',
        'Attwell': 'Stuart Attwell',
        'Madley': 'Andy Madley',
        'Gillett': 'Jarred Gillett',
        'Brooks': 'John Brooks',
        'Bankes': 'Peter Bankes',
        'Salisbury': 'Michael Salisbury',
        'England': 'Darren England',
        'Harrington': 'Tony Harrington',
        'Tierney': 'Paul Tierney',
        'Hooper': 'Simon Hooper',
        'Robinson': 'Tim Robinson',
        'Barrott': 'Samuel Barrott',
        'Jones': 'Robert Jones',
        'Bramall': 'Thomas Bramall',
        'Donohue': 'Matthew Donohue',
        'Kirk': 'Thomas Kirk',
    }

    def __init__(self, data_dir: str = "data/01-raw/premier_league/2025"):
        """
        Initialize the referee fetcher.

        Args:
            data_dir: Path to Premier League data directory
        """
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; BetTip/1.0)'
        })

    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name to match dataset conventions."""
        name = name.strip()
        return self.TEAM_ALIASES.get(name, name)

    def _parse_referee_equipment_page(self, html: str) -> List[Dict]:
        """
        Parse referee assignments from referee-equipment.com blog post.

        Returns:
            List of dicts with keys: home_team, away_team, referee, date
        """
        soup = BeautifulSoup(html, 'html.parser')
        assignments = []

        # Find the article content
        article = soup.find('article') or soup.find('div', class_='blog-content')
        if not article:
            article = soup

        # Get full text with spaces between elements
        text = article.get_text(separator=' ')

        # Split by date patterns
        date_pattern = r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d+(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})'
        chunks = re.split(date_pattern, text)

        # Known Premier League team names for validation
        known_teams = {
            'AFC Bournemouth', 'Arsenal', 'Aston Villa', 'Brentford', 'Brighton',
            'Brighton & Hove Albion', 'Burnley', 'Chelsea', 'Crystal Palace',
            'Everton', 'Fulham', 'Leeds', 'Leeds United', 'Leeds United AFC',
            'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle',
            'Newcastle United', 'Nottingham Forest', 'Southampton', 'Sunderland',
            'Sunderland AFC', 'Tottenham', 'Tottenham Hotspur', 'West Ham',
            'West Ham United', 'Wolverhampton Wanderers', 'Wolves'
        }

        # Process each date chunk
        for i in range(1, len(chunks), 2):
            date_str = chunks[i]
            content = chunks[i + 1] if i + 1 < len(chunks) else ''

            # Parse date
            date_match = re.match(
                r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+'
                r'(\d+)(?:st|nd|rd|th)?\s+'
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+'
                r'(\d{4})',
                date_str
            )
            if date_match:
                day, month_str, year = date_match.groups()
                month = datetime.strptime(month_str, '%B').month
                parsed_date = f"{year}-{month:02d}-{int(day):02d}"
            else:
                parsed_date = None

            # Find matches: Team1 v Team2Referee: RefName
            match_pattern = r'([A-Za-z\s&\.]+?)\s+v\s+([A-Za-z\s&\.]+?)Referee:\s*([A-Za-z,\s]+?)(?=Assistant|Fourth|$)'
            matches = re.findall(match_pattern, content)

            for home_raw, away_raw, ref_raw in matches:
                home = home_raw.strip()
                away = away_raw.strip()
                referee = ref_raw.strip()

                # Common assistant referee first names to remove
                noise_names = ['Marc', 'Sam', 'Stuart', 'Sian', 'Craig', 'Ian',
                              'Daniel', 'Simon', 'Tim', 'Nick', 'Paul', 'Gary',
                              'Adam', 'Thomas', 'James', 'Lee', 'Matthew', 'Wade',
                              'Adrian', 'Blake', 'Neil', 'Scott', 'Edward', 'Mark',
                              'Akil', 'Gavin', 'Steven', 'Constantine', 'Natalie']

                # Clean home team - remove leading noise names
                for noise in noise_names:
                    if home.startswith(noise + ' '):
                        home = home[len(noise):].strip()
                    # Also handle case like "Smith, Wade Liverpool" -> remove comma part
                    if ', ' + noise + ' ' in home:
                        home = home.split(', ' + noise + ' ')[-1].strip()

                # Clean away team
                for noise in noise_names:
                    if away.startswith(noise + ' '):
                        away = away[len(noise):].strip()
                    if ', ' + noise + ' ' in away:
                        away = away.split(', ' + noise + ' ')[-1].strip()

                # Normalize team names
                home = self._normalize_team_name(home)
                away = self._normalize_team_name(away)

                # Extract just referee surname
                # Format is "Surname, FirstName" -> get just surname
                if ',' in referee:
                    referee = referee.split(',')[0].strip()

                assignments.append({
                    'home_team': home,
                    'away_team': away,
                    'referee': referee,
                    'date': parsed_date
                })

        return assignments

    def fetch_matchweek_assignments(self, matchweek: int) -> List[Dict]:
        """
        Fetch referee assignments for a specific matchweek.

        Args:
            matchweek: Matchweek number (1-38)

        Returns:
            List of referee assignments
        """
        # Try to find the blog post URL for this matchweek
        url = f"{self.REFEREE_EQUIPMENT_BASE}/premier-league-referee-appointments-matchweek-{matchweek}-2025-26"

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return self._parse_referee_equipment_page(response.text)
            else:
                logger.warning(f"Could not fetch matchweek {matchweek}: HTTP {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"Error fetching matchweek {matchweek}: {e}")
            return []

    def fetch_recent_matchweeks(self, start: int = 18, end: int = 22) -> List[Dict]:
        """
        Fetch referee assignments for a range of recent matchweeks.

        Args:
            start: Starting matchweek
            end: Ending matchweek (inclusive)

        Returns:
            Combined list of all assignments
        """
        all_assignments = []

        for mw in range(start, end + 1):
            logger.info(f"Fetching matchweek {mw}...")
            assignments = self.fetch_matchweek_assignments(mw)
            all_assignments.extend(assignments)
            logger.info(f"  Found {len(assignments)} assignments")

        return all_assignments

    def update_fixtures_with_referees(
        self,
        assignments: List[Dict],
        matches_path: Optional[Path] = None
    ) -> Tuple[int, int]:
        """
        Update fixtures parquet file with referee assignments.

        Args:
            assignments: List of referee assignments from fetch
            matches_path: Path to matches.parquet (default: data_dir/matches.parquet)

        Returns:
            Tuple of (updated_count, total_upcoming)
        """
        if matches_path is None:
            matches_path = self.data_dir / "matches.parquet"

        if not matches_path.exists():
            raise FileNotFoundError(f"Matches file not found: {matches_path}")

        matches = pd.read_parquet(matches_path)

        # Only update upcoming matches (NS status)
        upcoming_mask = matches['fixture.status.short'] == 'NS'

        updated_count = 0

        for assignment in assignments:
            if not assignment.get('referee'):
                continue

            home = assignment['home_team']
            away = assignment['away_team']
            referee = assignment['referee']

            # Skip if team names have noise (contain assistant referee names)
            noise_names = {'Wade', 'Adrian', 'Blake', 'Marc', 'Sam', 'Stuart'}
            if any(n in home for n in noise_names) or any(n in away for n in noise_names):
                logger.debug(f"Skipping noisy assignment: {home} vs {away}")
                continue

            # Find matching fixture - try multiple matching strategies
            match_mask = None

            # Strategy 1: Exact match
            exact_mask = (
                upcoming_mask &
                (matches['teams.home.name'].str.lower() == home.lower()) &
                (matches['teams.away.name'].str.lower() == away.lower())
            )
            if exact_mask.any():
                match_mask = exact_mask

            # Strategy 2: Contains match (for partial names)
            if match_mask is None or not match_mask.any():
                contains_mask = (
                    upcoming_mask &
                    matches['teams.home.name'].str.contains(home, case=False, na=False, regex=False) &
                    matches['teams.away.name'].str.contains(away, case=False, na=False, regex=False)
                )
                if contains_mask.any():
                    match_mask = contains_mask

            # Strategy 3: Fuzzy match - home starts with our team name
            if match_mask is None or not match_mask.any():
                # Try matching just first word
                home_first = home.split()[0] if home else ''
                away_first = away.split()[0] if away else ''
                if home_first and away_first:
                    fuzzy_mask = (
                        upcoming_mask &
                        matches['teams.home.name'].str.startswith(home_first, na=False) &
                        matches['teams.away.name'].str.startswith(away_first, na=False)
                    )
                    if fuzzy_mask.any():
                        match_mask = fuzzy_mask

            if match_mask is not None and match_mask.any():
                matching = matches[match_mask]
                idx = matching.index[0]

                # Convert surname to full name
                full_name = self.REFEREE_NAMES.get(referee, referee)

                # Determine country (most PL refs are English, Gillett is Australian)
                country = 'Australia' if referee == 'Gillett' else 'England'

                matches.loc[idx, 'fixture.referee'] = f"{full_name}, {country}"
                updated_count += 1
                actual_home = matches.loc[idx, 'teams.home.name']
                actual_away = matches.loc[idx, 'teams.away.name']
                logger.info(f"Updated: {actual_home} vs {actual_away} -> {full_name}")
            else:
                logger.warning(f"No match found for: {home} vs {away}")

        # Save updated matches
        if updated_count > 0:
            matches.to_parquet(matches_path, index=False)
            logger.info(f"Saved {updated_count} updates to {matches_path}")

        total_upcoming = upcoming_mask.sum()
        return updated_count, total_upcoming


def main():
    """Fetch and update referee assignments."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("REFEREE ASSIGNMENT FETCHER")
    print("=" * 70)

    fetcher = RefereeFetcher()

    # Fetch recent matchweeks
    print("\nFetching referee assignments from referee-equipment.com...")
    assignments = fetcher.fetch_recent_matchweeks(start=20, end=22)

    print(f"\nFound {len(assignments)} assignments total")

    # Show sample
    print("\nSample assignments:")
    for a in assignments[:10]:
        print(f"  {a['home_team']} vs {a['away_team']} -> {a['referee']}")

    # Update fixtures
    print("\nUpdating fixtures with referee data...")
    updated, total = fetcher.update_fixtures_with_referees(assignments)

    print(f"\nResult: Updated {updated}/{total} upcoming fixtures with referee data")


if __name__ == "__main__":
    main()
