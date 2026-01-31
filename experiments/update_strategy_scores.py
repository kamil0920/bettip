#!/usr/bin/env python3
"""
Update strategy scores with actual match outcomes.

Reads strategy_scores.jsonl, fetches actual results for finished matches,
and writes strategy_results.jsonl with outcome (WON/LOST/NO_BET) and pnl per strategy.

Usage:
    python experiments/update_strategy_scores.py
    python experiments/update_strategy_scores.py --dry-run
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

SCORES_FILE = project_root / "data" / "06-prematch" / "strategy_scores.jsonl"
RESULTS_FILE = project_root / "data" / "06-prematch" / "strategy_results.jsonl"

# Market outcome evaluation functions
LEAGUE_IDS = {
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
}


def _infer_league(team_name: str) -> Optional[str]:
    """Infer league from team name (reuses logic from update_results.py)."""
    teams_by_league = {
        "premier_league": [
            "manchester", "liverpool", "arsenal", "chelsea", "tottenham",
            "newcastle", "brighton", "bournemouth", "fulham", "wolves",
            "brentford", "crystal", "everton", "west ham", "burnley",
            "nottingham", "aston", "leeds", "sunderland",
        ],
        "la_liga": [
            "barcelona", "real madrid", "atletico", "sevilla", "villarreal",
            "real sociedad", "athletic", "valencia", "betis", "celta",
            "getafe", "osasuna", "mallorca", "alaves", "girona",
        ],
        "serie_a": [
            "juventus", "inter", "milan", "napoli", "roma", "lazio",
            "atalanta", "fiorentina", "torino", "bologna", "sassuolo",
            "verona", "udinese", "lecce", "genoa", "cagliari",
        ],
        "bundesliga": [
            "bayern", "dortmund", "leverkusen", "leipzig", "frankfurt",
            "hoffenheim", "wolfsburg", "freiburg", "union", "stuttgart",
            "gladbach", "mainz", "augsburg", "werder", "bochum", "heidenheim",
        ],
        "ligue_1": [
            "psg", "paris", "marseille", "lyon", "monaco", "lille",
            "nice", "lens", "rennes", "strasbourg", "nantes",
            "montpellier", "toulouse", "reims", "le havre", "auxerre",
        ],
    }
    team_lower = team_name.lower()
    for league, teams in teams_by_league.items():
        for t in teams:
            if t in team_lower:
                return league
    return None


def _load_fixtures(league: str, season: int = 2025) -> pd.DataFrame:
    """Load fixtures from local parquet."""
    path = project_root / f"data/01-raw/{league}/{season}/matches.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_match_stats(league: str, season: int = 2025) -> pd.DataFrame:
    """Load match statistics from local parquet."""
    path = project_root / f"data/01-raw/{league}/{season}/match_stats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _find_fixture(
    fixtures_df: pd.DataFrame, home: str, away: str, date: str
) -> Optional[Dict]:
    """Find fixture by team names and date."""
    if fixtures_df.empty:
        return None

    def normalize(name: str) -> str:
        return name.lower().replace(" ", "").replace("-", "").replace(".", "")[:10]

    home_n = normalize(home)
    away_n = normalize(away)

    for _, row in fixtures_df.iterrows():
        if "teams.home.name" in row.index:
            rh = str(row.get("teams.home.name", ""))
            ra = str(row.get("teams.away.name", ""))
            fd = str(row.get("fixture.date", ""))[:10]
            fid = row.get("fixture.id")
            hg = row.get("goals.home")
            ag = row.get("goals.away")
            status = row.get("fixture.status.short", "")
        else:
            rh = str(row.get("home_team", ""))
            ra = str(row.get("away_team", ""))
            fd = str(row.get("date", ""))[:10]
            fid = row.get("fixture_id")
            hg = row.get("home_goals")
            ag = row.get("away_goals")
            status = row.get("status", "")

        if normalize(rh).startswith(home_n) or home_n in normalize(rh):
            if normalize(ra).startswith(away_n) or away_n in normalize(ra):
                if fd == date[:10]:
                    return {
                        "fixture_id": fid,
                        "home_goals": int(hg) if pd.notna(hg) else None,
                        "away_goals": int(ag) if pd.notna(ag) else None,
                        "status": status,
                    }
    return None


def evaluate_market(
    market: str, fixture_info: Dict, stats: Optional[Dict], line: float = 0.0
) -> Optional[Tuple[bool, float]]:
    """
    Evaluate whether a market outcome was a win.

    Returns (won: bool, actual_value: float) or None if match not finished.
    """
    if fixture_info.get("status") not in ("FT", "AET", "PEN"):
        return None

    hg = fixture_info.get("home_goals")
    ag = fixture_info.get("away_goals")
    if hg is None or ag is None:
        return None

    total_goals = hg + ag

    if market == "home_win":
        return (hg > ag, float(hg - ag))
    elif market == "away_win":
        return (ag > hg, float(ag - hg))
    elif market == "over25":
        return (total_goals > 2.5, float(total_goals))
    elif market == "under25":
        return (total_goals < 2.5, float(total_goals))
    elif market == "btts":
        return (hg > 0 and ag > 0, float(int(hg > 0 and ag > 0)))

    # Niche markets need stats
    if stats is None:
        return None

    home_stats = stats.get("home", {})
    away_stats = stats.get("away", {})

    if market == "fouls" or market.startswith("fouls_"):
        total = (home_stats.get("fouls", 0) or 0) + (away_stats.get("fouls", 0) or 0)
        return (total > line, float(total))
    elif market == "shots" or market.startswith("shots_"):
        total = (home_stats.get("shots", 0) or 0) + (away_stats.get("shots", 0) or 0)
        return (total > line, float(total))
    elif market == "corners" or market.startswith("corners_"):
        total = (home_stats.get("corner_kicks", 0) or 0) + (
            away_stats.get("corner_kicks", 0) or 0
        )
        return (total > line, float(total))

    return None


def update_scores(dry_run: bool = False) -> int:
    """Read strategy_scores.jsonl, resolve outcomes, write strategy_results.jsonl."""
    if not SCORES_FILE.exists():
        print(f"No scores file found: {SCORES_FILE}")
        return 0

    # Load existing results to avoid re-processing
    existing_keys = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing_keys.add((r["fixture_id"], r["market"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    # Caches
    fixtures_cache: Dict[str, pd.DataFrame] = {}
    stats_cache: Dict[str, pd.DataFrame] = {}
    today = datetime.now().strftime("%Y-%m-%d")

    updated = 0
    results_to_write = []

    with open(SCORES_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = (entry.get("fixture_id", ""), entry.get("market", ""))
            if key in existing_keys:
                continue

            # Skip future matches
            if entry.get("date", "") >= today:
                continue

            league = entry.get("league") or _infer_league(entry.get("home", ""))
            if not league:
                continue

            # Load fixtures
            if league not in fixtures_cache:
                fixtures_cache[league] = _load_fixtures(league)
            if league not in stats_cache:
                stats_cache[league] = _load_match_stats(league)

            fixture_info = _find_fixture(
                fixtures_cache[league], entry["home"], entry["away"], entry["date"]
            )
            if not fixture_info:
                continue

            # Load stats for niche markets
            stats = None
            fid = fixture_info.get("fixture_id")
            if fid and not stats_cache[league].empty:
                stats_df = stats_cache[league]
                if "fixture_id" in stats_df.columns:
                    match_row = stats_df[stats_df["fixture_id"] == fid]
                    if not match_row.empty:
                        row = match_row.iloc[0]
                        stats = {
                            "home": {
                                "fouls": int(row.get("home_fouls", 0) or 0),
                                "shots": int(row.get("home_shots", 0) or 0),
                                "corner_kicks": int(row.get("home_corners", 0) or 0),
                            },
                            "away": {
                                "fouls": int(row.get("away_fouls", 0) or 0),
                                "shots": int(row.get("away_shots", 0) or 0),
                                "corner_kicks": int(row.get("away_corners", 0) or 0),
                            },
                        }

            market = entry["market"]
            # For niche strategies, parse line from strategy name
            result_entry = dict(entry)
            result_entry["fixture_status"] = fixture_info.get("status")
            result_entry["home_goals"] = fixture_info.get("home_goals")
            result_entry["away_goals"] = fixture_info.get("away_goals")

            # Evaluate each strategy
            for strat_name, strat_data in entry.get("strategies", {}).items():
                # Determine line for niche markets from strategy name
                line = 0.0
                if market in ("fouls", "shots", "corners"):
                    parts = strat_name.split("_")
                    for i, p in enumerate(parts):
                        if p == "over" and i + 1 < len(parts):
                            try:
                                line = float("_".join(parts[i + 1:]).replace("_", "."))
                            except ValueError:
                                pass
                            break

                outcome = evaluate_market(market, fixture_info, stats, line)
                if outcome is None:
                    continue

                won, actual = outcome
                passed = strat_data.get("pass", False)

                if passed:
                    strat_data["outcome"] = "WON" if won else "LOST"
                    strat_data["pnl"] = round(
                        (strat_data.get("edge", 0) / 100.0) if won else -1.0, 3
                    )
                else:
                    strat_data["outcome"] = "NO_BET"
                    strat_data["pnl"] = 0.0

                strat_data["actual"] = actual

            result_entry["strategies"] = entry["strategies"]
            results_to_write.append(result_entry)
            updated += 1

    if not dry_run and results_to_write:
        with open(RESULTS_FILE, "a") as f:
            for r in results_to_write:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {updated} results to {RESULTS_FILE}")
    elif dry_run:
        print(f"[DRY RUN] Would write {updated} results")

    return updated


def main():
    parser = argparse.ArgumentParser(description="Update strategy scores with outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    updated = update_scores(dry_run=args.dry_run)
    print(f"Total entries updated: {updated}")


if __name__ == "__main__":
    main()
