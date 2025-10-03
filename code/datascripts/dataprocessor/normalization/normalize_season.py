"""
normalize_season.py (patched, robust)

Normalize raw fixtures_detailed_all_<season>.parquet into:
 - matches.parquet
 - events.parquet
 - player_stats.parquet
 - teams.parquet

This version is defensive about NaNs, JSON-string fields, and logs row-level errors
instead of crashing.
"""
import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import sys
import traceback

# Imports from modules
from helpers import sanitize_for_write
from extractors import extract_fixture_core, extract_events_from_row, extract_player_stats_from_row
from normalizers import normalize_player_stats_df
from process_player_stats import build_player_form_features


# ---------- merge helper ----------

def load_detailed_file(season_dir: Path, season: int):
    merged = season_dir / f"fixtures_detailed_all_{season}.parquet"
    if merged.exists():
        return merged
    batch_files = sorted(season_dir.glob("fixtures_detailed_batch_*.parquet"))
    if not batch_files:
        return None
    dfs = []
    bad = []
    for f in batch_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            bad.append((f.name, str(e)))
    if bad:
        print("Some batch files failed to read:", bad)
    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = sanitize_for_write(df_all)
    df_all.to_parquet(merged, index=False)
    return merged


# ---------- main normalization ----------

def normalize_season(season: int, base_dir: Path):
    season_dir = base_dir / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")

    print(f"Loading merged detailed file for season {season} ...")
    merged_path = load_detailed_file(season_dir, season)
    if merged_path is None:
        raise FileNotFoundError(
            "No merged detailed file found (fixtures_detailed_all_<season>.parquet or batch files).")
    print("Reading", merged_path)
    df = pd.read_parquet(merged_path)
    print("Rows (fixtures):", len(df))

    matches_rows = []
    events_rows = []
    players_rows = []
    teams_map = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Normalizing season {season}"):
        try:
            # copy and aggressively parse JSON-like string columns into Python objects
            row_parsed = row.copy()
            for col in row.index:
                val = row.get(col)
                if isinstance(val, str):
                    s = val.strip()
                    if s and s[0] in ('{', '[') and s[-1] in ('}', ']'):
                        try:
                            row_parsed[col] = json.loads(val)
                        except Exception:
                            try:
                                row_parsed[col] = json.loads(val.encode('utf-8').decode('unicode_escape'))
                            except Exception:
                                # leave as-is
                                pass

            # extract match
            m = extract_fixture_core(row_parsed)
            matches_rows.append(m)

            # collect teams
            if m.get("home_team_id"):
                teams_map[m["home_team_id"]] = teams_map.get(m["home_team_id"], {"team.id": m["home_team_id"],
                                                                                 "team.name": m.get("home_team_name")})
            if m.get("away_team_id"):
                teams_map[m["away_team_id"]] = teams_map.get(m["away_team_id"], {"team.id": m["away_team_id"],
                                                                                 "team.name": m.get("away_team_name")})

            # events
            evs = extract_events_from_row(row_parsed)
            if evs:
                events_rows.extend(evs)

            # players
            pls = extract_player_stats_from_row(row_parsed)
            if pls:
                players_rows.extend(pls)

        except Exception as err:
            # log and continue
            print(f"[normalize] Error processing row idx={idx}: {err}")
            traceback.print_exc()
            # also save a small JSON snippet to debug file
            try:
                excerpt_path = season_dir / f"normalize_error_row_{season}_{idx}.json"
                row_dict = {c: (str(row[c])[:1000] if not pd.isna(row[c]) else None) for c in row.index}
                excerpt_path.write_text(json.dumps(row_dict), encoding='utf-8')
                print("Wrote debug snippet to", excerpt_path)
            except Exception:
                pass
            continue

    matches_df = pd.DataFrame(matches_rows).drop_duplicates(subset=["fixture_id"])
    events_df = pd.DataFrame(events_rows) if events_rows else pd.DataFrame(
        columns=["fixture_id", "minute", "extra", "type", "detail", "team_id", "player_id", "assist_id", "raw"])
    players_df = pd.DataFrame(players_rows) if players_rows else pd.DataFrame(
        columns=["fixture_id", "player_id", "player_name", "team_id", "minutes", "goals", "assists", "yellow_cards",
                 "red_cards", "raw"])
    players_normalized = normalize_player_stats_df(players_df)

    teams_df = pd.DataFrame(list(teams_map.values())) if teams_map else pd.DataFrame(columns=["team.id", "team.name"])

    # coerce numeric columns safely
    for col in ["fixture_id", "home_team_id", "away_team_id", "venue_id", "league_id", "ft_home", "ft_away", "ht_home",
                "ht_away", "timestamp"]:
        if col in matches_df.columns:
            matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce')
    for col in ["fixture_id", "team_id", "player_id", "assist_id", "minute", "extra", "minutes", "goals", "assists",
                "yellow_cards", "red_cards"]:
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors='coerce')
        if col in players_normalized.columns:
            players_normalized[col] = pd.to_numeric(players_normalized[col], errors='coerce')

    # set Int64 for integer-like columns
    for col in ["fixture_id", "home_team_id", "away_team_id", "venue_id", "league_id", "ft_home", "ft_away", "ht_home",
                "ht_away", "timestamp"]:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].astype('Int64')

    processed_player_features = build_player_form_features(
        players_normalized, matches_df, keep_lead_cols=True, drop_old_features=True
    )

    out_matches = season_dir / "matches.parquet"
    out_events = season_dir / "events.parquet"
    out_players = season_dir / "player_stats.parquet"
    out_teams = season_dir / "teams.parquet"
    out_players_processed = season_dir / "processed_player_stats.parquet"

    print("Writing matches:", out_matches)
    matches_df.to_parquet(out_matches, index=False)
    print("Writing events:", out_events, " rows:", len(events_df))
    events_df.to_parquet(out_events, index=False)
    print("Writing player_stats:", out_players, " rows:", len(players_normalized))
    players_normalized.to_parquet(out_players, index=False)
    print("Writing teams:", out_teams, " rows:", len(teams_df))
    teams_df.to_parquet(out_teams, index=False)
    print("Writing processed_player_stats:", out_players_processed, " rows:", len(processed_player_features))
    processed_player_features.to_parquet(out_players_processed, index=False)


    print("Normalization complete for season", season)
    return {
        "matches": out_matches,
        "events": out_events,
        "players": out_players,
        "teams": out_teams
    }


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="Season year (e.g. 2025)")
    ap.add_argument("--base-dir", type=str, default="data/seasons", help="Base dir for season folders")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        normalize_season(args.season, Path(args.base_dir))
    except Exception as e:
        print("FATAL ERROR:", e)
        traceback.print_exc()
        sys.exit(2)
