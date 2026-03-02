"""Overlay real bookmaker per-line odds into training features.

Loads historical per-line odds from two sources:
1. Sportmonks CSV exports (Big 5 leagues, 2022-2026, corners + cards)
2. The Odds API historical parquet (Big 5 + expansion, Feb 2026+, corners + cards)

Normalizes team names to match features parquet, and merges via (league, date,
home_norm, away_norm).  Matches without coverage keep NaN and will be filled
later by the NB CDF estimator in per_line_odds.py.
"""

import logging
import unicodedata
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Sportmonks league_id → features-parquet league name
LEAGUE_ID_MAP: Dict[int, str] = {
    8: "premier_league",
    82: "bundesliga",
    301: "ligue_1",
    384: "serie_a",
    564: "la_liga",
}

# Target lines we actually train on (ignore extreme Sportmonks lines)
TARGET_LINES: Dict[str, list] = {
    "corners": [8.5, 9.5, 10.5, 11.5],
    "cards": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
}

# Static alias table: lowercased + diacritics-stripped Sportmonks name → features name.
# Applied to BOTH sides so all representations of the same team converge.
TEAM_ALIASES: Dict[str, str] = {
    # EPL
    "newcastle united": "newcastle",
    "west ham united": "west ham",
    "sheffield united": "sheffield utd",
    "west bromwich albion": "west brom",
    "norwich city": "norwich",
    "afc bournemouth": "bournemouth",
    "brighton & hove albion": "brighton",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
    # Bundesliga
    "borussia mgladbach": "borussia monchengladbach",  # M'gladbach apostrophe stripped
    "bayern munchen": "bayern munich",
    "fc bayern munchen": "bayern munich",
    "dsc arminia bielefeld": "arminia bielefeld",
    "hertha bsc": "hertha berlin",
    "mainz 05": "fsv mainz 05",
    "stuttgart": "vfb stuttgart",
    "tsg hoffenheim": "1899 hoffenheim",
    "hoffenheim": "1899 hoffenheim",
    "vfl bochum 1848": "vfl bochum",
    "wolfsburg": "vfl wolfsburg",
    "fc koln": "1 fc koln",
    "augsburg": "fc augsburg",
    "bayer 04 leverkusen": "bayer leverkusen",
    "fc union berlin": "union berlin",
    "freiburg": "sc freiburg",
    "heidenheim": "fc heidenheim",
    "schalke 04": "fc schalke 04",
    "st pauli": "fc st pauli",
    # La Liga
    "celta de vigo": "celta vigo",
    "deportivo alaves": "alaves",
    "real oviedo": "oviedo",
    "real valladolid": "valladolid",
    "sd eibar": "eibar",
    "fc barcelona": "barcelona",
    "granada": "granada cf",
    # Ligue 1
    "angers sco": "angers",
    "brest": "stade brestois 29",
    "clermont": "clermont foot",
    "olympique lyonnais": "lyon",
    "troyes": "estac troyes",
    "losc lille": "lille",
    "olympique marseille": "marseille",
    "paris": "paris saint germain",
    "saint-etienne": "saint etienne",
    # Serie A
    "hellas verona": "verona",
    "roma": "as roma",
    "milan": "ac milan",
}

# Default data directories
SPORTMONKS_DATA_DIR = Path("data/sportmonks_odds/processed")
THEODDS_HISTORICAL_PATH = Path("data/historical_niche_odds/niche_odds_historical.parquet")


def _normalize_team(name: str) -> str:
    """Aggressive normalization for team name matching.

    Lowercase, strip diacritics, remove punctuation, apply alias table.
    """
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    # Strip diacritics (ö→o, ü→u, é→e, etc.)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # Remove apostrophes and dots
    s = s.replace("'", "").replace(".", "")
    # Collapse whitespace
    s = " ".join(s.split())
    # Apply alias table
    return TEAM_ALIASES.get(s, s)


def _line_to_col_suffix(line: float) -> str:
    """Convert line value to column suffix: 1.5 -> '15', 8.5 -> '85'."""
    return str(int(line * 10))


def load_sportmonks_per_line_odds(
    data_dir: Path = SPORTMONKS_DATA_DIR,
) -> pd.DataFrame:
    """Load and pivot Sportmonks per-line odds for corners and cards.

    Returns a DataFrame with columns:
        [league, date, home_norm, away_norm, {stat}_{direction}_avg_{suffix}, ...]
    Only includes our target lines (corners 8.5-11.5, cards 1.5-6.5).
    """
    frames = []

    for stat, csv_name in [("corners", "corners_odds.csv"), ("cards", "cards_odds.csv")]:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"Sportmonks {stat} CSV not found: {csv_path}")
            continue

        raw = pd.read_csv(csv_path)
        target_lines = TARGET_LINES[stat]

        # Filter to target lines only
        raw = raw[raw["line"].isin(target_lines)].copy()
        if raw.empty:
            logger.warning(f"No rows for target lines in {csv_name}")
            continue

        # Map league_id → league name, drop unknown
        raw["league"] = raw["league_id"].map(LEAGUE_ID_MAP)
        raw = raw.dropna(subset=["league"])

        # Extract date from start_time
        raw["date"] = pd.to_datetime(raw["start_time"]).dt.date

        # Normalize team names
        raw["home_norm"] = raw["home_team"].apply(_normalize_team)
        raw["away_norm"] = raw["away_team"].apply(_normalize_team)

        # Pivot each line into over/under columns
        for line in target_lines:
            suffix = _line_to_col_suffix(line)
            over_col = f"{stat}_over_avg_{suffix}"
            under_col = f"{stat}_under_avg_{suffix}"

            line_rows = raw[raw["line"] == line].copy()
            if line_rows.empty:
                continue

            # Deduplicate: keep row with highest bookmaker count (most reliable average)
            line_rows = line_rows.sort_values("over_count", ascending=False)
            line_rows = line_rows.drop_duplicates(
                subset=["league", "date", "home_norm", "away_norm"], keep="first"
            )

            line_rows = line_rows.rename(
                columns={"over_avg": over_col, "under_avg": under_col}
            )
            keep_cols = ["league", "date", "home_norm", "away_norm", over_col, under_col]
            frames.append(line_rows[keep_cols])

    if not frames:
        logger.warning("No Sportmonks per-line odds loaded")
        return pd.DataFrame()

    # Merge all line-pivoted frames on the match key
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(
            f, on=["league", "date", "home_norm", "away_norm"], how="outer"
        )

    logger.info(
        f"Sportmonks per-line odds: {len(result)} fixtures, "
        f"{len([c for c in result.columns if '_avg_' in c])} odds columns"
    )
    return result


def load_theodds_historical_odds(
    parquet_path: Path = THEODDS_HISTORICAL_PATH,
) -> pd.DataFrame:
    """Load The Odds API historical niche odds and pivot to wide format.

    Returns a DataFrame with the same schema as load_sportmonks_per_line_odds():
        [league, date, home_norm, away_norm, {stat}_{direction}_avg_{suffix}, ...]

    Only includes corners and cards at our target lines.
    """
    if not parquet_path.exists():
        logger.warning(f"The Odds API historical parquet not found: {parquet_path}")
        return pd.DataFrame()

    raw = pd.read_parquet(parquet_path)

    # Filter to corners and cards only (skip cornershc/cardshc — different column scheme)
    raw = raw[raw["market"].isin(["corners", "cards"])].copy()
    if raw.empty:
        logger.warning("No corners/cards rows in The Odds API historical data")
        return pd.DataFrame()

    # Filter to target lines
    masks = []
    for stat, lines in TARGET_LINES.items():
        mask = (raw["market"] == stat) & (raw["line"].isin(lines))
        masks.append(mask)
    raw = raw[masks[0] | masks[1]].copy()
    if raw.empty:
        logger.warning("No rows at target lines in The Odds API historical data")
        return pd.DataFrame()

    # Extract date, normalize teams
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw["home_norm"] = raw["home_team"].apply(_normalize_team)
    raw["away_norm"] = raw["away_team"].apply(_normalize_team)

    # Pivot each stat × line into over/under columns
    frames = []
    for stat, lines in TARGET_LINES.items():
        stat_rows = raw[raw["market"] == stat]
        for line in lines:
            suffix = _line_to_col_suffix(line)
            over_col = f"{stat}_over_avg_{suffix}"
            under_col = f"{stat}_under_avg_{suffix}"

            line_rows = stat_rows[stat_rows["line"] == line].copy()
            if line_rows.empty:
                continue

            # Deduplicate: keep row with highest bookmaker count
            line_rows = line_rows.sort_values("num_bookmakers", ascending=False)
            line_rows = line_rows.drop_duplicates(
                subset=["league", "date", "home_norm", "away_norm"], keep="first"
            )

            line_rows = line_rows.rename(
                columns={"over_avg": over_col, "under_avg": under_col}
            )
            keep_cols = ["league", "date", "home_norm", "away_norm", over_col, under_col]
            frames.append(line_rows[keep_cols])

    if not frames:
        logger.warning("No The Odds API historical odds after pivoting")
        return pd.DataFrame()

    # Merge all line-pivoted frames on the match key
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(
            f, on=["league", "date", "home_norm", "away_norm"], how="outer"
        )

    logger.info(
        f"The Odds API historical odds: {len(result)} fixtures, "
        f"{len([c for c in result.columns if '_avg_' in c])} odds columns"
    )
    return result


def _combine_odds_sources(sources: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple odds DataFrames, deduplicating on match key.

    When both sources have data for the same fixture × line, keeps the row
    with more non-NaN odds columns (proxy for higher bookmaker count).
    """
    if len(sources) == 1:
        return sources[0]

    match_key = ["league", "date", "home_norm", "away_norm"]

    # Align columns across sources — fill missing odds columns with NaN
    all_odds_cols = set()
    for s in sources:
        all_odds_cols.update(c for c in s.columns if "_avg_" in c)

    aligned = []
    for s in sources:
        for col in all_odds_cols:
            if col not in s.columns:
                s[col] = pd.NA
        aligned.append(s)

    combined = pd.concat(aligned, ignore_index=True)

    # Count non-NaN odds columns per row as quality score
    odds_cols = [c for c in combined.columns if "_avg_" in c]
    combined["_n_odds"] = combined[odds_cols].notna().sum(axis=1)

    # Deduplicate: keep row with most non-NaN odds columns
    combined = combined.sort_values("_n_odds", ascending=False)
    combined = combined.drop_duplicates(subset=match_key, keep="first")
    combined = combined.drop(columns=["_n_odds"])

    logger.info(f"Combined odds: {len(combined)} fixtures from {len(sources)} sources")
    return combined


def overlay_sportmonks_per_line_odds(
    df: pd.DataFrame,
    data_dir: Path = SPORTMONKS_DATA_DIR,
    theodds_path: Path = THEODDS_HISTORICAL_PATH,
) -> pd.DataFrame:
    """Overlay real bookmaker per-line odds into a features DataFrame.

    Loads from both Sportmonks CSVs and The Odds API historical parquet,
    concatenates them (preferring higher bookmaker count on duplicates),
    then merges into the features DataFrame.

    Matches on (league, date, home_norm, away_norm). Only writes to cells
    that are NaN or where the column doesn't yet exist — never overwrites
    existing values (e.g. from live The Odds API).

    Args:
        df: Features DataFrame with 'league', 'date', 'home_team_name',
            'away_team_name' columns.
        data_dir: Path to Sportmonks processed CSVs.
        theodds_path: Path to The Odds API historical parquet.

    Returns:
        DataFrame with bookmaker odds overlaid where available.
    """
    required = {"league", "date", "home_team_name", "away_team_name"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning(f"Missing columns for per-line odds overlay: {missing}")
        return df

    sm_odds = load_sportmonks_per_line_odds(data_dir)
    theodds = load_theodds_historical_odds(theodds_path)

    # Combine both sources
    if sm_odds.empty and theodds.empty:
        return df

    sources = [s for s in [sm_odds, theodds] if not s.empty]
    combined = _combine_odds_sources(sources)
    if combined.empty:
        return df

    sm_odds = combined

    # Normalize features team names using the same function
    df["_home_norm"] = df["home_team_name"].apply(_normalize_team)
    df["_away_norm"] = df["away_team_name"].apply(_normalize_team)
    df["_date"] = pd.to_datetime(df["date"]).dt.date

    # Build merge key
    sm_odds["date"] = pd.to_datetime(sm_odds["date"])
    if hasattr(sm_odds["date"].iloc[0], "date"):
        sm_odds["date"] = sm_odds["date"].dt.date

    odds_cols = [c for c in sm_odds.columns if "_avg_" in c]

    merged = df.merge(
        sm_odds,
        left_on=["league", "_date", "_home_norm", "_away_norm"],
        right_on=["league", "date", "home_norm", "away_norm"],
        how="left",
        suffixes=("", "_sm"),
    )

    # Overlay: for each odds column, fill NaN in df with Sportmonks values
    n_filled = 0
    for col in odds_cols:
        sm_col = f"{col}_sm" if f"{col}_sm" in merged.columns else col
        if sm_col not in merged.columns:
            continue

        if col in df.columns:
            # Only fill where original is NaN
            mask = merged[col].isna() & merged[sm_col].notna()
            if mask.any():
                merged.loc[mask, col] = merged.loc[mask, sm_col]
                n_filled += mask.sum()
        else:
            # Column doesn't exist yet — create it from Sportmonks
            merged[col] = merged[sm_col]
            n_filled += merged[col].notna().sum()

    # Drop temp/merge columns
    drop_cols = ["_home_norm", "_away_norm", "_date", "home_norm", "away_norm"]
    drop_cols += [c for c in merged.columns if c.endswith("_sm")]
    # Also drop the Sportmonks 'date' if it duplicated (date_sm or similar)
    drop_cols += [c for c in merged.columns if c == "date_sm"]
    drop_cols = [c for c in drop_cols if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    n_matched = (
        df.set_index(["league", "_date", "_home_norm", "_away_norm"])
        .index.isin(
            sm_odds.set_index(["league", "date", "home_norm", "away_norm"]).index
        )
        .sum()
    )

    # Clean up temp columns from original df
    df.drop(columns=["_home_norm", "_away_norm", "_date"], inplace=True, errors="ignore")

    logger.info(
        f"Sportmonks overlay: {n_matched}/{len(df)} fixtures matched, "
        f"{n_filled} cells filled across {len(odds_cols)} columns"
    )

    return merged
