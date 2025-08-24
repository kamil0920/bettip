import os
import pandas as pd
from api_client import ApiFootballClient

client = ApiFootballClient()

def get_league_by_name(name="Premier League", country="England"):
    # /leagues?name=Premier%20League&country=England
    resp = client.get("/leagues", params={"name": name, "country": country})
    return resp

def get_teams_for_season(league_id=39, season=2023):
    resp = client.get("/teams", params={"league": league_id, "season": season})
    return resp

def save_json_as_parquet(data, path):
    df = pd.json_normalize(data)
    df.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")

if __name__ == "__main__":
    # 1) find league
    leagues = get_league_by_name()
    print("Leagues response keys:", list(leagues.keys()))
    # inspect results
    if leagues.get("response"):
        df_leagues = pd.json_normalize(leagues["response"])
        df_leagues.to_parquet("leagues.parquet", index=False)
        print("Saved leagues.parquet")
    else:
        print("No league data returned")

    # 2) teams for season
    season = 2023  # change as needed
    teams = get_teams_for_season(league_id=39, season=season)
    if teams.get("response"):
        save_json_as_parquet(teams["response"], f"teams_{season}.parquet")
    else:
        print("No teams data")

    # 3) optionally fetch venues from team objects
    # many team responses include 'venue' nested object
    teams_df = pd.json_normalize(teams["response"])
    # extract venue objects if present
    if "venue.name" in teams_df.columns or "venue.city" in teams_df.columns:
        venue_cols = [c for c in teams_df.columns if c.startswith("venue.")]
        venues_df = teams_df[venue_cols].rename(columns=lambda c: c.replace("venue.", ""))
        venues_df.to_parquet("venues.parquet", index=False)
        print("Saved venues.parquet")
    else:
        print("No venue object found in teams response")
