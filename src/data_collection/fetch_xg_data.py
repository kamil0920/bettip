#!/usr/bin/env python3
"""
Fetch real xG (Expected Goals) data from Understat.

Understat provides free xG data for:
- Premier League (EPL)
- La Liga
- Bundesliga
- Serie A
- Ligue 1

This matches our 5 target leagues.
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from understat import Understat

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# League mapping: our names -> Understat names
LEAGUE_MAP = {
    'premier_league': 'EPL',
    'la_liga': 'La_Liga',
    'bundesliga': 'Bundesliga',
    'serie_a': 'Serie_A',
    'ligue_1': 'Ligue_1',
}

# Season mapping: our format -> Understat format
# Understat uses the starting year (e.g., 2024 for 2024-25 season)
SEASON_MAP = {
    '2019': 2019,
    '2020': 2020,
    '2021': 2021,
    '2022': 2022,
    '2023': 2023,
    '2024': 2024,
    '2025': 2025,
}


async def fetch_league_matches(session, league: str, season: int) -> list:
    """Fetch all matches with xG data for a league and season."""
    try:
        understat = Understat(session)
        matches = await understat.get_league_results(league, season)
        return matches
    except Exception as e:
        print(f"  Error fetching {league} {season}: {e}")
        return []


async def fetch_all_xg_data(leagues: list = None, seasons: list = None) -> pd.DataFrame:
    """Fetch xG data for all specified leagues and seasons."""
    import aiohttp

    if leagues is None:
        leagues = list(LEAGUE_MAP.keys())
    if seasons is None:
        seasons = [2019, 2020, 2021, 2022, 2023, 2024]

    all_matches = []

    async with aiohttp.ClientSession() as session:
        for our_league, understat_league in LEAGUE_MAP.items():
            if our_league not in leagues:
                continue

            for season in seasons:
                print(f"Fetching {our_league} {season}...")
                matches = await fetch_league_matches(session, understat_league, season)

                if matches:
                    for match in matches:
                        all_matches.append({
                            'league': our_league,
                            'season': season,
                            'match_id': match.get('id'),
                            'date': match.get('datetime', ''),
                            'home_team': match.get('h', {}).get('title', ''),
                            'away_team': match.get('a', {}).get('title', ''),
                            'home_goals': int(match.get('goals', {}).get('h', 0)),
                            'away_goals': int(match.get('goals', {}).get('a', 0)),
                            'home_xg': float(match.get('xG', {}).get('h', 0)),
                            'away_xg': float(match.get('xG', {}).get('a', 0)),
                            'is_finished': match.get('isResult', False),
                        })
                    print(f"  Found {len(matches)} matches")
                else:
                    print(f"  No data found")

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

    df = pd.DataFrame(all_matches)
    return df


def normalize_team_name(name: str) -> str:
    """Normalize team names for matching - handles both directions."""
    if pd.isna(name):
        return ''

    name = str(name).strip()

    # Comprehensive mappings (Understat name -> standard name)
    # We'll normalize both sides to a common format
    understat_to_standard = {
        # Premier League
        'West Ham United': 'West Ham',
        'Tottenham Hotspur': 'Tottenham',
        'Tottenham': 'Tottenham',
        'Wolverhampton Wanderers': 'Wolves',
        'Newcastle United': 'Newcastle',
        'Brighton and Hove Albion': 'Brighton',
        'Brighton & Hove Albion': 'Brighton',
        'Sheffield United': 'Sheffield Utd',
        'Leeds United': 'Leeds',
        'Leicester City': 'Leicester',
        'Nottingham Forest': 'Nottingham Forest',
        "Nott'ham Forest": 'Nottingham Forest',
        'Luton Town': 'Luton',
        'Ipswich Town': 'Ipswich',

        # Bundesliga
        'Bayern Munich': 'Bayern München',
        'Borussia M.Gladbach': 'Borussia Mönchengladbach',
        'Borussia Monchengladbach': 'Borussia Mönchengladbach',
        'FC Cologne': '1. FC Köln',
        'Cologne': '1. FC Köln',
        'Hertha Berlin': 'Hertha BSC',
        'Greuther Furth': 'SpVgg Greuther Fürth',
        'Arminia Bielefeld': 'Arminia Bielefeld',
        'Union Berlin': 'Union Berlin',
        'Hoffenheim': '1899 Hoffenheim',
        'Wolfsburg': 'VfL Wolfsburg',
        'Bochum': 'Vfl Bochum',
        'VfL Bochum': 'Vfl Bochum',
        'Mainz 05': 'FSV Mainz 05',
        'Mainz': 'FSV Mainz 05',
        'Freiburg': 'SC Freiburg',
        'Augsburg': 'FC Augsburg',
        'Heidenheim': '1. FC Heidenheim',
        'FC Heidenheim': '1. FC Heidenheim',
        'Darmstadt 98': 'SV Darmstadt 98',
        'Darmstadt': 'SV Darmstadt 98',
        'Holstein Kiel': 'Holstein Kiel',
        'St. Pauli': 'FC St. Pauli',

        # Serie A
        'Inter Milan': 'Inter',
        'Internazionale': 'Inter',
        'Roma': 'AS Roma',
        'Hellas Verona': 'Verona',
        'SPAL 2013': 'SPAL',
        'Spezia': 'Spezia',
        'Salernitana': 'Salernitana',

        # La Liga
        'Atletico Madrid': 'Atletico Madrid',
        'Athletic Bilbao': 'Athletic Club',
        'Celta Vigo': 'Celta Vigo',
        'Cadiz': 'Cadiz CF',
        'Mallorca': 'RCD Mallorca',
        'Espanyol': 'Espanyol',
        'Getafe': 'Getafe',
        'Granada': 'Granada CF',
        'Valladolid': 'Valladolid',
        'Real Valladolid': 'Valladolid',
        'Elche': 'Elche',
        'Leganes': 'Leganes',
        'Alaves': 'Alaves',
        'Deportivo Alaves': 'Alaves',
        'Las Palmas': 'UD Las Palmas',

        # Ligue 1
        'Paris Saint-Germain': 'Paris Saint Germain',
        'PSG': 'Paris Saint Germain',
        'Olympique Lyonnais': 'Lyon',
        'Olympique Lyon': 'Lyon',
        'Olympique Marseille': 'Marseille',
        'AS Monaco': 'Monaco',
        'LOSC Lille': 'Lille',
        'Stade Rennais': 'Rennes',
        'RC Strasbourg': 'Strasbourg',
        'RC Strasbourg Alsace': 'Strasbourg',
        'Montpellier HSC': 'Montpellier',
        'Stade Brestois': 'Brest',
        'Stade Brest': 'Brest',
        'Stade Brestois 29': 'Brest',
        'FC Nantes': 'Nantes',
        'Angers SCO': 'Angers',
        'RC Lens': 'Lens',
        'Saint-Etienne': 'Saint Etienne',
        'AS Saint-Etienne': 'Saint Etienne',
        'Estac Troyes': 'Troyes',
        'ESTAC Troyes': 'Troyes',
        'FC Metz': 'Metz',
        'Stade de Reims': 'Reims',
        'OGC Nice': 'Nice',
        'FC Toulouse': 'Toulouse',
        'AJ Auxerre': 'Auxerre',
        'Le Havre AC': 'Le Havre',
        'Le Havre': 'Le Havre',
    }

    return understat_to_standard.get(name, name)


def merge_xg_with_features(xg_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Merge xG data with existing features."""
    xg_df = xg_df.copy()
    features_df = features_df.copy()

    # Normalize dates
    xg_df['date'] = pd.to_datetime(xg_df['date']).dt.date
    features_df['date'] = pd.to_datetime(features_df['date']).dt.date

    # Normalize team names in BOTH dataframes
    xg_df['home_team_norm'] = xg_df['home_team'].apply(normalize_team_name)
    xg_df['away_team_norm'] = xg_df['away_team'].apply(normalize_team_name)

    # Determine which column to use for feature team names
    home_col = 'home_team_name' if 'home_team_name' in features_df.columns else 'home_team'
    away_col = 'away_team_name' if 'away_team_name' in features_df.columns else 'away_team'

    # Normalize feature team names too
    features_df['home_team_norm'] = features_df[home_col].apply(normalize_team_name)
    features_df['away_team_norm'] = features_df[away_col].apply(normalize_team_name)

    # Create merge key
    xg_df['merge_key'] = xg_df['date'].astype(str) + '_' + xg_df['home_team_norm'] + '_' + xg_df['away_team_norm']
    features_df['merge_key'] = features_df['date'].astype(str) + '_' + features_df['home_team_norm'] + '_' + features_df['away_team_norm']

    # Merge
    xg_cols = ['merge_key', 'home_xg', 'away_xg']
    merged = features_df.merge(
        xg_df[xg_cols].drop_duplicates('merge_key'),
        on='merge_key',
        how='left'
    )

    # Calculate xG-based features
    merged['real_xg_diff'] = merged['home_xg'] - merged['away_xg']
    merged['real_xg_total'] = merged['home_xg'] + merged['away_xg']
    merged['real_xg_home_ratio'] = merged['home_xg'] / (merged['real_xg_total'] + 0.001)

    # Calculate xG vs actual goals (for historical matches)
    if 'home_goals' in merged.columns:
        merged['home_xg_overperform'] = merged['home_goals'] - merged['home_xg']
        merged['away_xg_overperform'] = merged['away_goals'] - merged['away_xg']

    # Drop temp columns
    merged = merged.drop(columns=['merge_key', 'home_team_norm', 'away_team_norm'], errors='ignore')

    return merged


async def main():
    """Main function to fetch and save xG data."""
    print("=" * 60)
    print("FETCHING XG DATA FROM UNDERSTAT")
    print("=" * 60)

    # Fetch data for all leagues and recent seasons
    xg_df = await fetch_all_xg_data(
        leagues=['premier_league', 'bundesliga', 'serie_a', 'la_liga', 'ligue_1'],
        seasons=[2019, 2020, 2021, 2022, 2023, 2024, 2025]
    )

    print(f"\nTotal matches fetched: {len(xg_df)}")

    # Save raw xG data
    output_path = project_root / 'data/03-features/xg_data_understat.csv'
    xg_df.to_csv(output_path, index=False)
    print(f"Saved xG data to: {output_path}")

    # Show sample
    print("\nSample data:")
    print(xg_df[['league', 'date', 'home_team', 'away_team', 'home_xg', 'away_xg']].head(10).to_string())

    # Coverage stats
    print("\nCoverage by league:")
    print(xg_df.groupby('league').size())

    # Try to merge with existing features
    features_path = project_root / 'data/03-features/features_all_leagues_complete.csv'
    if features_path.exists():
        print("\nMerging with existing features...")
        from src.utils.data_io import load_features
        features_df = load_features(features_path)
        merged_df = merge_xg_with_features(xg_df, features_df)

        # Check merge success rate
        xg_matched = merged_df['home_xg'].notna().sum()
        total = len(merged_df)
        print(f"xG match rate: {xg_matched}/{total} ({xg_matched/total:.1%})")

        # Save merged features
        merged_path = project_root / 'data/03-features/features_with_real_xg.csv'
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged features to: {merged_path}")

    return xg_df


if __name__ == '__main__':
    asyncio.run(main())
