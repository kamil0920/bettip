"""Normalization functions for player stats and performance metrics."""

import json
import pandas as pd
import numpy as np
from helpers import first_notna, maybe_load_json


def normalize_player_stats_df(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand players_df['raw'] JSON into columns. Robust to:
      - raw being a JSON string or dict
      - 'statistics' being missing, empty, dict or list
      - retains fixture_id/player_id/team_id
    Returns normalized dataframe with statistics flattened and numeric coercion applied.
    """
    if players_df is None or len(players_df) == 0:
        # return empty canonical frame (same file schema you want to write)
        return pd.DataFrame(columns=[
            "fixture_id", "player_id", "player_name", "team_id",
            "minutes", "goals", "assists", "yellow_cards", "red_cards", "raw"
        ])

    # local helper to parse raw
    def parse_raw(r):
        if isinstance(r, str):
            try:
                return maybe_load_json(r)
            except Exception:
                try:
                    return json.loads(r)
                except Exception:
                    return r
        return r

    df = players_df.copy().reset_index(drop=True)
    df["raw_obj"] = df["raw"].apply(parse_raw)

    # extract statistics list for each row in a consistent shape (list of dicts)
    def extract_stats_list(obj):
        if obj is None:
            return []
        if isinstance(obj, dict):
            if "statistics" in obj and isinstance(obj["statistics"], list):
                return obj["statistics"]
            if "players" in obj and isinstance(obj["players"], list):
                return obj.get("statistics") or []
            if any(k in obj for k in ("games","goals","cards","minutes")):
                return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []

    df["stats_list"] = df["raw_obj"].apply(extract_stats_list)

    # explode so we have one row per stat item (if stats_list empty we'll keep one row with NaN stat)
    df["stats_list"] = df["stats_list"].apply(lambda L: L if (isinstance(L, list) and len(L) > 0) else [None])
    df_exp = df.explode("stats_list").reset_index(drop=True)

    # normalize the stats_list dict into columns
    stats_norm = pd.json_normalize(df_exp["stats_list"].apply(lambda x: x or {}))
    # merge base columns with normalized stats
    out = pd.concat([df_exp.drop(columns=["raw_obj", "stats_list"]), stats_norm], axis=1)

    # some stats pack goals/assists under nested dicts. try common patterns:
    def safe_get_nested(series, *path):
        cur = series
        for k in path:
            try:
                if cur is None:
                    return None
                cur = cur.get(k) if isinstance(cur, dict) else None
            except Exception:
                return None
        return cur

    # Create canonical stat columns if present
    out["minutes"] = out.apply(
        lambda r: first_notna(
            safe_get_nested(r.get("stats_list", {}), "games", "minutes"),
            r.get("minutes"),
            safe_get_nested(r.get("raw_obj", {}), "games", "minutes")
        ), axis=1
    )

    out["goals"] = out.apply(
        lambda r: first_notna(
            safe_get_nested(r.get("stats_list", {}), "goals", "total"),
            r.get("goals"),
            safe_get_nested(r.get("raw_obj", {}), "goals", "total"),
            r.get("goals_total"),
        ), axis=1
    )

    out["assists"] = out.apply(
        lambda r: first_notna(
            safe_get_nested(r.get("stats_list", {}), "goals", "assists"),
            r.get("assists"),
            safe_get_nested(r.get("raw_obj", {}), "goals", "assists"),
        ), axis=1
    )

    out["yellow_cards"] = out.apply(
        lambda r: first_notna(
            safe_get_nested(r.get("stats_list", {}), "cards", "yellow"),
            r.get("yellow"),
            r.get("yellow_cards"),
            safe_get_nested(r.get("raw_obj", {}), "cards", "yellow"),
        ), axis=1
    )
    out["red_cards"] = out.apply(
        lambda r: first_notna(
            safe_get_nested(r.get("stats_list", {}), "cards", "red"),
            r.get("red"),
            r.get("red_cards"),
            safe_get_nested(r.get("raw_obj", {}), "cards", "red"),
        ), axis=1
    )

    # retain identifying cols
    for col in ("fixture_id","player_id","player_name","team_id","raw"):
        if col not in out.columns and col in players_df.columns:
            out[col] = players_df[col].values.repeat(df["stats_list"].apply(len)).tolist()[:len(out)]

    # fix column names: replace dots with underscore
    out.columns = [str(c).replace(".", "_") for c in out.columns]

    # coerce numeric fields cleanly
    for ncol in ("minutes","goals","assists","yellow_cards","red_cards","fixture_id","player_id","team_id"):
        if ncol in out.columns:
            out[ncol] = pd.to_numeric(out[ncol], errors="coerce")

    # Set integer NA-supporting dtypes for id columns
    for icol in ("fixture_id","player_id","team_id"):
        if icol in out.columns:
            out[icol] = out[icol].astype("Int64")

    # Fill missing count stats with 0
    for c in ("goals","assists","minutes","yellow_cards","red_cards"):
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    # Sensible column order
    cols_order = ["fixture_id","player_id","player_name","team_id","games_rating","minutes","goals","assists","yellow_cards","red_cards","raw"]
    remaining = [c for c in out.columns if c not in cols_order]
    out = out[[c for c in cols_order if c in out.columns] + remaining]

    return out


def normalize_stats_per_90_by_position(players_df: pd.DataFrame, drop_old_features=False) -> pd.DataFrame:
    """
    Normalize key performance stats by 90 minutes separately by player position group.
    Adds a new column 'performance_per_90' as a combined metric per position.
    Optionally drops old features to reduce noise.

    Args:
    players_df: DataFrame with 'games_position', 'minutes', and stats columns.
    drop_old_features: If True, retain only essential columns.

    Returns:
    DataFrame with 'performance_per_90' added (and olds dropped if specified).
    """

    df = players_df.copy()

    # Function to calculate performance for each group
    def calc_performance(row):
        pos = row.get('games_position')
        mins = row.get('minutes', 0)
        if mins == 0 or mins is None or pd.isna(mins):
            return 0.0

        # Group stats differently by position
        if pos == 'G':  # Goalkeepers: focus on saves, goals_conceded
            saves = row.get('goals_saves', 0) or 0
            goals_conceded = row.get('goals_conceded', 0) or 0
            # Normalize saves minus goals conceded per 90
            return ((saves - goals_conceded) / mins) * 90
        elif pos == 'D':  # Defenders: tackles, interceptions, duels_won
            tackles = row.get('tackles_total', 0) or 0
            interceptions = row.get('tackles_interceptions', 0) or 0
            duels = row.get('duels_won', 0) or 0
            return ((tackles + interceptions + duels) / mins) * 90
        elif pos == 'M':  # Midfielders: passes (total, key), assists
            passes = row.get('passes_total', 0) or 0
            key_passes = row.get('passes_key', 0) or 0
            assists = row.get('assists', 0) or 0
            return ((passes + key_passes + 3*assists) / mins) * 90
        elif pos == 'F':  # Forwards: goals, shots, dribbles
            goals = row.get('goals', 0) or 0
            shots = row.get('shots_total', 0) or 0
            dribbles = row.get('dribbles_success', 0) or 0
            return ((4*goals + shots + 2*dribbles) / mins) * 90
        else:
            # Unknown position or missing, simple sum of goals + assists normalized
            goals = row.get('goals', 0) or 0
            assists = row.get('assists', 0) or 0
            return ((goals + assists) / mins) * 90

    # Create 'performance_per_90' column
    df['performance_per_90'] = df.apply(calc_performance, axis=1).fillna(0)

    if drop_old_features:
        # Define columns to keep (customize as needed)
        keep_cols = [
            'fixture_id', 'player_id', 'player_name', 'team_id', 'games_position',
            'minutes', 'games_rating', 'raw', 'performance_per_90'
        ]
        df = df[[col for col in keep_cols if col in df.columns]]

    return df
