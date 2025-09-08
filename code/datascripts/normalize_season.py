#!/usr/bin/env python3
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
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import traceback


# ---------- helpers ----------

def first_notna(*vals):
    """Return first value from vals that is not None and not NaN/NA. Keeps 0."""
    for v in vals:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        return v
    return None


def maybe_load_json(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        if (s[0] in ('{', '[')) and (s[-1] in ('}', ']')):
            try:
                return json.loads(x)
            except Exception:
                # attempt unescape
                try:
                    un = x.encode('utf-8').decode('unicode_escape')
                    return json.loads(un)
                except Exception:
                    return x
    return x


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s and s[0] == '[' and s[-1] == ']':
            try:
                parsed = json.loads(x)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def to_int_safe(v):
    """Convert v to int or return None. Handles NaN/pd.NA/empty strings."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    # if already int-like
    try:
        return int(v)
    except Exception:
        pass
    # try parsing numeric string like "123.0"
    try:
        s = str(v).strip()
        if s == "":
            return None
        # remove percent if present (not expected here but safe)
        s2 = s.replace("%", "")
        return int(float(s2))
    except Exception:
        return None


def get_nested_from_obj(obj, *keys):
    cur = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


# ---------- extraction helpers ----------

def extract_fixture_core(series):
    """Extract core match info from a row (pandas Series)."""

    def try_keys(*keys):
        dotted = ".".join(keys)
        if dotted in series.index and pd.notna(series.get(dotted)):
            v = series.get(dotted)
            if isinstance(v, str):
                return maybe_load_json(v)
            return v
        if keys[0] in series.index:
            val = series.get(keys[0])
            if isinstance(val, str):
                val = maybe_load_json(val)
            if isinstance(val, dict):
                return get_nested_from_obj(val, *keys[1:]) if len(keys) > 1 else val
        # scan other dict-like columns
        for c in series.index:
            v = series.get(c)
            if isinstance(v, str):
                v = maybe_load_json(v)
            if isinstance(v, dict):
                cur = v
                found = True
                for k in keys:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        found = False
                        break
                if found:
                    return cur
        return None

    fixture_id = first_notna(
        try_keys("fixture", "id"),
        series.get("fixture.id"),
        series.get("id")
    )

    date = first_notna(
        try_keys("fixture", "date"),
        series.get("fixture.date"),
        series.get("date")
    )

    timestamp = first_notna(
        try_keys("fixture", "timestamp"),
        series.get("fixture.timestamp"),
        series.get("timestamp")
    )

    referee = first_notna(
        try_keys("fixture", "referee"),
        series.get("fixture.referee"),
        series.get("referee")
    )

    venue_id = first_notna(
        try_keys("fixture", "venue", "id"),
        series.get("fixture.venue.id"),
        series.get("venue.id")
    )

    venue_name = first_notna(
        try_keys("fixture", "venue", "name"),
        series.get("fixture.venue.name"),
        series.get("venue.name")
    )

    status = first_notna(
        try_keys("fixture", "status", "short"),
        series.get("fixture.status.short"),
        series.get("status")
    )

    league_id = first_notna(
        try_keys("league", "id"),
        series.get("league.id"),
        series.get("league_id")
    )

    round_name = first_notna(
        try_keys("league", "round"),
        series.get("league.round"),
        series.get("round")
    )

    home_id = first_notna(
        try_keys("teams", "home", "id"),
        series.get("teams.home.id")
    )
    home_name = first_notna(
        try_keys("teams", "home", "name"),
        series.get("teams.home.name")
    )
    away_id = first_notna(
        try_keys("teams", "away", "id"),
        series.get("teams.away.id")
    )
    away_name = first_notna(
        try_keys("teams", "away", "name"),
        series.get("teams.away.name")
    )

    ft_home = first_notna(
        try_keys("score", "fulltime", "home"),
        try_keys("goals", "home"),
        series.get("goals.home"),
        series.get("goals_home")
    )
    ft_away = first_notna(
        try_keys("score", "fulltime", "away"),
        try_keys("goals", "away"),
        series.get("goals.away"),
        series.get("goals_away")
    )
    ht_home = first_notna(
        try_keys("score", "halftime", "home"),
        series.get("score.halftime.home"),
        series.get("ht_home")
    )
    ht_away = first_notna(
        try_keys("score", "halftime", "away"),
        series.get("score.halftime.away"),
        series.get("ht_away")
    )

    return {
        "fixture_id": to_int_safe(fixture_id),
        "date": date,
        "timestamp": to_int_safe(timestamp),
        "referee": referee,
        "venue_id": to_int_safe(venue_id),
        "venue_name": venue_name,
        "status": status,
        "league_id": to_int_safe(league_id),
        "round": round_name,
        "home_team_id": to_int_safe(home_id),
        "home_team_name": home_name,
        "away_team_id": to_int_safe(away_id),
        "away_team_name": away_name,
        "ft_home": to_int_safe(ft_home),
        "ft_away": to_int_safe(ft_away),
        "ht_home": to_int_safe(ht_home),
        "ht_away": to_int_safe(ht_away),
    }


def extract_events_from_row(series):
    events_found = []
    candidates = []

    # explicit 'events' col
    if 'events' in series.index:
        v = series.get('events')
        if isinstance(v, str):
            v = maybe_load_json(v)
        if isinstance(v, list):
            candidates.append(v)
        elif isinstance(v, dict):
            # dict of lists?
            for val in v.values():
                if isinstance(val, list):
                    candidates.append(val)

    # scan other columns for list-of-dicts
    for c in series.index:
        v = series.get(c)
        if isinstance(v, str):
            v = maybe_load_json(v)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            candidates.append(v)

    fixture_id = series.get('fixture_id') or series.get('fixture.id')
    for cand in candidates:
        for ev in ensure_list(cand):
            if not isinstance(ev, dict):
                continue
            minute = None
            extra = None
            time_obj = ev.get('time') or ev.get('minute') or ev.get('elapsed')
            if isinstance(time_obj, dict):
                minute = time_obj.get('elapsed') or time_obj.get('minute')
                extra = time_obj.get('extra')
            else:
                minute = time_obj
            etype = ev.get('type') or ev.get('event') or ev.get('result') or ev.get('detail')
            edetail = ev.get('detail') or ev.get('description') or ev.get('comment')
            team = ev.get('team') or {}
            team_id = team.get('id') if isinstance(team, dict) else None
            player = ev.get('player') or {}
            player_id = None
            if isinstance(player, dict):
                player_id = player.get('id') or player.get('player_id')
            else:
                player_id = ev.get('player_id') or ev.get('playerId')
            assist = ev.get('assist') or {}
            assist_id = assist.get('id') if isinstance(assist, dict) else None

            events_found.append({
                "fixture_id": to_int_safe(fixture_id),
                "minute": to_int_safe(minute),
                "extra": to_int_safe(extra),
                "type": etype,
                "detail": edetail,
                "team_id": to_int_safe(team_id),
                "player_id": to_int_safe(player_id),
                "assist_id": to_int_safe(assist_id),
                "raw": json.dumps(ev, default=str)
            })
    return events_found


def extract_player_stats_from_row(series):
    out = []
    # preferred 'players' column
    if 'players' in series.index:
        val = series.get('players')
        if isinstance(val, str):
            val = maybe_load_json(val)
        if isinstance(val, list):
            for block in val:
                if not isinstance(block, dict):
                    continue
                team_obj = block.get('team') or {}
                team_id = team_obj.get('id') if isinstance(team_obj, dict) else None
                players_block = block.get('players') or block.get('statistics') or block.get('players')
                if isinstance(players_block, str):
                    players_block = maybe_load_json(players_block)
                if isinstance(players_block, list):
                    for p in players_block:
                        player_meta = p.get('player') if isinstance(p, dict) else None
                        stats = p.get('statistics') or p.get('stats') or p.get('statistics')
                        if isinstance(stats, dict):
                            stats = [stats]
                        if isinstance(stats, list) and stats:
                            st = stats[0]
                        else:
                            st = p
                        player_id = None
                        player_name = None
                        if isinstance(player_meta, dict):
                            player_id = player_meta.get('id') or player_meta.get('player_id')
                            player_name = player_meta.get('name')
                        else:
                            player_id = p.get('player_id') or p.get('id')
                            player_name = p.get('player_name') or p.get('name')
                        minutes = None
                        goals = None
                        assists = None
                        yellow = None
                        red = None
                        if isinstance(st, dict):
                            minutes = get_nested_from_obj(st, 'games', 'minutes') or st.get('minutes')
                            goals = get_nested_from_obj(st, 'goals', 'total') if isinstance(st.get('goals'),
                                                                                            dict) else st.get('goals')
                            assists = get_nested_from_obj(st, 'goals', 'assists') if isinstance(st.get('goals'),
                                                                                                dict) else st.get(
                                'assists')
                            yellow = get_nested_from_obj(st, 'cards', 'yellow') if isinstance(st.get('cards'),
                                                                                              dict) else st.get(
                                'yellow')
                            red = get_nested_from_obj(st, 'cards', 'red') if isinstance(st.get('cards'),
                                                                                        dict) else st.get('red')
                        out.append({
                            "fixture_id": to_int_safe(series.get('fixture_id') or series.get('fixture.id')),
                            "player_id": to_int_safe(player_id),
                            "player_name": player_name,
                            "team_id": to_int_safe(team_id),
                            "minutes": to_int_safe(minutes),
                            "goals": to_int_safe(goals) if goals is not None else 0,
                            "assists": to_int_safe(assists),
                            "yellow_cards": to_int_safe(yellow),
                            "red_cards": to_int_safe(red),
                            "raw": json.dumps(p, default=str)
                        })
    else:
        # fallback scan
        for c in series.index:
            v = series.get(c)
            if isinstance(v, str):
                v = maybe_load_json(v)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                sample = v[0]
                if 'player' in sample or 'player_id' in sample or 'statistics' in sample:
                    for p in v:
                        player_id = None
                        player_name = None
                        if isinstance(p.get('player'), dict):
                            player_id = p['player'].get('id')
                            player_name = p['player'].get('name')
                        else:
                            player_id = p.get('player_id') or p.get('id')
                            player_name = p.get('player_name') or p.get('name')
                        goals = p.get('goals') or (
                            p.get('statistics', [{}])[0].get('goals') if p.get('statistics') else None)
                        yellow = p.get('yellow') or (p.get('cards', {}).get('yellow') if p.get('cards') else None)
                        out.append({
                            "fixture_id": to_int_safe(series.get('fixture_id') or series.get('fixture.id')),
                            "player_id": to_int_safe(player_id),
                            "player_name": player_name,
                            "team_id": None,
                            "minutes": None,
                            "goals": to_int_safe(goals),
                            "assists": None,
                            "yellow_cards": to_int_safe(yellow),
                            "red_cards": None,
                            "raw": json.dumps(p, default=str)
                        })
    return out


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


def sanitize_for_write(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    def safe_val(v):
        if v is None: return None
        try:
            if pd.isna(v): return None
        except Exception:
            pass
        if isinstance(v, (str, int, float, bool)): return v
        if isinstance(v, (list, dict, tuple, np.ndarray)):
            try:
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                return json.dumps(v, default=str)
            except Exception:
                return str(v)
        return str(v)

    for col in df.columns:
        if df[col].dtype == 'O':
            sample = df[col].head(50)
            need = False
            for val in sample:
                if val is None:
                    continue
                if not isinstance(val, (str, int, float, bool)):
                    need = True
                    break
            if need:
                df[col] = df[col].apply(safe_val)
    return df


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
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce')

    # set Int64 for integer-like columns
    for col in ["fixture_id", "home_team_id", "away_team_id", "venue_id", "league_id", "ft_home", "ft_away", "ht_home",
                "ht_away", "timestamp"]:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].astype('Int64')

    out_matches = season_dir / "matches.parquet"
    out_events = season_dir / "events.parquet"
    out_players = season_dir / "player_stats.parquet"
    out_teams = season_dir / "teams.parquet"

    print("Writing matches:", out_matches)
    matches_df.to_parquet(out_matches, index=False)
    print("Writing events:", out_events, " rows:", len(events_df))
    events_df.to_parquet(out_events, index=False)
    print("Writing player_stats:", out_players, " rows:", len(players_df))
    players_df.to_parquet(out_players, index=False)
    print("Writing teams:", out_teams, " rows:", len(teams_df))
    teams_df.to_parquet(out_teams, index=False)

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
