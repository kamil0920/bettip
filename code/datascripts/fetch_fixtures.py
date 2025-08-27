import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from api_client import ApiFootballClient

def fetch_fixtures_list(client, league_id, season):
    """Fetch all fixtures (basic info) for a given season."""
    data = client.get("/fixtures", params={"league": league_id, "season": season})
    return data.get("response", [])

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def fetch_fixture_details(client, fixture_ids, batch_size=20, limit_batches=None):
    """
    Fetch detailed fixtures data in batches of up to 20 IDs.
    limit_batches: if set, limit to N batches per run (handy for rate-limit).
    """
    all_details = []
    batches = list(chunked(fixture_ids, batch_size))
    if limit_batches:
        batches = batches[:limit_batches]
    for batch in tqdm(batches, desc="Fetching fixture details"):
        ids_str = "-".join(str(fid) for fid in batch)
        data = client.get("/fixtures", params={"ids": ids_str})
        all_details.extend(data.get("response", []))
    return all_details

if __name__ == "__main__":
    client = ApiFootballClient()
    league_id = 39
    seasons = [2019, 2020, 2021, 2022, 2023]

    for season in seasons:
        # save under data/seasons/<season>/
        season_dir = Path(__file__).resolve().parent.parent / "data" / "seasons" / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Season {season}: Fetching fixture list ===")
        fixtures = fetch_fixtures_list(client, league_id, season)
        df = pd.json_normalize(fixtures)
        path_list = season_dir / f"fixtures_list_{season}.parquet"
        df.to_parquet(path_list, index=False)
        print(f"Saved {len(df)} fixtures to {path_list}")

        fixture_ids = df.get('fixture.id', [])
        details = fetch_fixture_details(client, fixture_ids, batch_size=20, limit_batches=5)
        if details:
            df_details = pd.json_normalize(details)
            path_details = season_dir / f"fixtures_detailed_part_{season}.parquet"
            df_details.to_parquet(path_details, index=False)
            print(f"Saved {len(df_details)} detailed fixtures to {path_details}")
