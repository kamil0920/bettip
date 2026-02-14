#!/usr/bin/env python3
"""
Update match results for pending betting recommendations.

Loads all rec_*.csv files from data/05-recommendations/ recursively,
identifies pending bets (result is empty/UNKNOWN/NaN), fetches match
results from API-Football, determines win/loss based on market type,
and updates the CSV files in place.

Usage:
    python scripts/update_results.py                           # Update all pending
    python scripts/update_results.py --dry-run                 # Preview without saving
    python scripts/update_results.py --date-range 2026-01-14 2026-01-20
    python scripts/update_results.py --verbose                 # Debug logging
"""
import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection.api_client import FootballAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RECOMMENDATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "05-recommendations"

# Maximum API calls per minute (safety margin below the 10/min tier limit)
MAX_CALLS_PER_MINUTE = 10
CALL_INTERVAL_SECONDS = 60.0 / MAX_CALLS_PER_MINUTE


def clean_val(val: Any, default: int = 0) -> int:
    """Clean API statistic value to int."""
    if val is None:
        return default
    if isinstance(val, str):
        val = val.replace("%", "").strip()
        try:
            return int(val) if val else default
        except (ValueError, TypeError):
            return default
    return int(val) if isinstance(val, (int, float)) else default


def parse_fixture_response(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse API-Football /fixtures response into a flat result dict.

    Returns None if the match is not finished or response is invalid.
    """
    fixtures = response.get("response", [])
    if not fixtures:
        return None

    fixture_data = fixtures[0]
    fixture_info = fixture_data.get("fixture", {})
    status = fixture_info.get("status", {}).get("short", "")

    # Only process finished matches
    if status not in ("FT", "AET", "PEN"):
        return None

    goals = fixture_data.get("goals", {})
    home_goals = goals.get("home")
    away_goals = goals.get("away")

    if home_goals is None or away_goals is None:
        return None

    result: Dict[str, Any] = {
        "fixture_id": fixture_info.get("id"),
        "status": status,
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "total_goals": int(home_goals) + int(away_goals),
    }

    # Parse statistics if available
    statistics = fixture_data.get("statistics", [])
    if statistics and len(statistics) >= 2:
        home_stats = {
            s["type"]: s["value"] for s in statistics[0].get("statistics", [])
        }
        away_stats = {
            s["type"]: s["value"] for s in statistics[1].get("statistics", [])
        }

        result["home_shots_on_target"] = clean_val(home_stats.get("Shots on Goal"))
        result["away_shots_on_target"] = clean_val(away_stats.get("Shots on Goal"))
        result["total_shots_on_target"] = (
            result["home_shots_on_target"] + result["away_shots_on_target"]
        )

        result["home_shots"] = clean_val(home_stats.get("Total Shots"))
        result["away_shots"] = clean_val(away_stats.get("Total Shots"))
        result["total_shots"] = result["home_shots"] + result["away_shots"]

        result["home_fouls"] = clean_val(home_stats.get("Fouls"))
        result["away_fouls"] = clean_val(away_stats.get("Fouls"))
        result["total_fouls"] = result["home_fouls"] + result["away_fouls"]

        result["home_corners"] = clean_val(home_stats.get("Corner Kicks"))
        result["away_corners"] = clean_val(away_stats.get("Corner Kicks"))
        result["total_corners"] = result["home_corners"] + result["away_corners"]

        result["home_yellow"] = clean_val(home_stats.get("Yellow Cards"))
        result["away_yellow"] = clean_val(away_stats.get("Yellow Cards"))
        result["home_red"] = clean_val(home_stats.get("Red Cards"))
        result["away_red"] = clean_val(away_stats.get("Red Cards"))
        result["total_cards"] = (
            result["home_yellow"]
            + result["away_yellow"]
            + result["home_red"]
            + result["away_red"]
        )

    return result


def determine_outcome(
    market: str,
    bet_type: str,
    line: float,
    match_result: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float]]:
    """Determine bet outcome (W/L) and actual value from match result.

    Args:
        market: Market type (HOME_WIN, AWAY_WIN, OVER_2.5, CORNERS, etc.)
        bet_type: Bet direction (OVER, UNDER, HOME_WIN, AWAY_WIN, etc.)
        line: Betting line value
        match_result: Parsed match result dict from parse_fixture_response

    Returns:
        Tuple of (result_str, actual_value). result_str is "W", "L", or None
        if outcome cannot be determined. actual_value is the relevant stat.
    """
    market_upper = market.upper().replace(" ", "_") if isinstance(market, str) else ""
    bet_upper = bet_type.upper().replace(" ", "_") if isinstance(bet_type, str) else ""

    home_goals = match_result["home_goals"]
    away_goals = match_result["away_goals"]
    total_goals = match_result["total_goals"]

    # --- Goal-based markets ---
    if market_upper in ("HOME_WIN", "MATCH_RESULT") and bet_upper in ("HOME_WIN", "HOME"):
        won = home_goals > away_goals
        return ("W" if won else "L"), float(home_goals)

    if market_upper in ("AWAY_WIN", "MATCH_RESULT") and bet_upper in ("AWAY_WIN", "AWAY"):
        won = away_goals > home_goals
        return ("W" if won else "L"), float(away_goals)

    if market_upper == "OVER_2.5":
        won = total_goals > 2.5
        return ("W" if won else "L"), float(total_goals)

    if market_upper == "UNDER_2.5":
        won = total_goals < 2.5
        return ("W" if won else "L"), float(total_goals)

    if market_upper == "BTTS":
        both_scored = home_goals > 0 and away_goals > 0
        if bet_upper in ("YES", "BTTS", "OVER"):
            return ("W" if both_scored else "L"), float(int(both_scored))
        elif bet_upper in ("NO", "UNDER"):
            return ("W" if not both_scored else "L"), float(int(both_scored))

    # --- Statistics-based markets (need stats from API) ---
    # Shots (total shots or shots on target depending on line context)
    if market_upper == "SHOTS":
        total_shots = match_result.get("total_shots")
        if total_shots is None:
            return None, None
        actual = float(total_shots)
        if bet_upper == "OVER":
            return ("W" if total_shots > line else "L"), actual
        elif bet_upper == "UNDER":
            return ("W" if total_shots < line else "L"), actual

    # Fouls
    if market_upper in ("FOULS", "FOULS_OVER", "FOULS_UNDER"):
        total_fouls = match_result.get("total_fouls")
        if total_fouls is None:
            return None, None
        actual = float(total_fouls)
        if bet_upper == "OVER" or market_upper == "FOULS_OVER":
            return ("W" if total_fouls > line else "L"), actual
        elif bet_upper == "UNDER" or market_upper == "FOULS_UNDER":
            return ("W" if total_fouls < line else "L"), actual

    # Corners
    if market_upper == "CORNERS":
        total_corners = match_result.get("total_corners")
        if total_corners is None:
            return None, None
        actual = float(total_corners)
        if bet_upper == "OVER":
            return ("W" if total_corners > line else "L"), actual
        elif bet_upper == "UNDER":
            return ("W" if total_corners < line else "L"), actual

    # Cards (yellow + red)
    if market_upper == "CARDS":
        total_cards = match_result.get("total_cards")
        if total_cards is None:
            return None, None
        actual = float(total_cards)
        if bet_upper == "OVER":
            return ("W" if total_cards > line else "L"), actual
        elif bet_upper == "UNDER":
            return ("W" if total_cards < line else "L"), actual

    # Asian handicap (basic support)
    if market_upper == "ASIAN_HANDICAP":
        if bet_upper == "HOME":
            adjusted = home_goals + line - away_goals
            won = adjusted > 0
            return ("W" if won else "L"), float(home_goals - away_goals)
        elif bet_upper == "AWAY":
            adjusted = away_goals + line - home_goals
            won = adjusted > 0
            return ("W" if won else "L"), float(away_goals - home_goals)

    logger.warning(f"Unknown market/bet_type combination: market={market}, bet_type={bet_type}")
    return None, None


def is_pending(result_val: Any) -> bool:
    """Check if a result value represents a pending/unsettled bet."""
    if pd.isna(result_val):
        return True
    result_str = str(result_val).strip().upper()
    return result_str in ("", "UNKNOWN", "NAN", "NONE", "PENDING")


def is_past_date(date_str: str) -> bool:
    """Check if a date string represents a date in the past."""
    try:
        match_date = pd.to_datetime(date_str).date()
        return match_date < date.today()
    except (ValueError, TypeError):
        return False


def load_recommendation_files(
    base_dir: Path,
    date_range: Optional[Tuple[str, str]] = None,
) -> List[Tuple[Path, pd.DataFrame]]:
    """Load all rec_*.csv files recursively from base_dir.

    Returns list of (file_path, dataframe) tuples.
    """
    csv_files = sorted(base_dir.rglob("rec_*.csv"))
    logger.info(f"Found {len(csv_files)} recommendation CSV files under {base_dir}")

    results: List[Tuple[Path, pd.DataFrame]] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, dtype=str)
            if df.empty:
                continue

            # Normalize column names to handle minor variations
            df.columns = df.columns.str.strip().str.lower()

            # Apply date range filter if specified
            if date_range and "date" in df.columns:
                start_date, end_date = date_range
                df_dates = pd.to_datetime(df["date"], errors="coerce")
                mask = (df_dates >= pd.to_datetime(start_date)) & (
                    df_dates <= pd.to_datetime(end_date)
                )
                if not mask.any():
                    continue
            elif date_range and "start_time" in df.columns:
                start_date, end_date = date_range
                df_dates = pd.to_datetime(df["start_time"], errors="coerce")
                mask = (df_dates.dt.date >= pd.to_datetime(start_date).date()) & (
                    df_dates.dt.date <= pd.to_datetime(end_date).date()
                )
                if not mask.any():
                    continue

            results.append((csv_path, df))
        except Exception as e:
            logger.warning(f"Failed to load {csv_path}: {e}")

    logger.info(f"Loaded {len(results)} files with data")
    return results


def get_date_column(df: pd.DataFrame) -> Optional[str]:
    """Determine which column holds the match date."""
    for col in ("date", "start_time"):
        if col in df.columns:
            return col
    return None


def get_result_column(df: pd.DataFrame) -> str:
    """Determine which column holds the result (W/L/UNKNOWN)."""
    for col in ("result", "status"):
        if col in df.columns:
            return col
    return "result"


def get_actual_column(df: pd.DataFrame) -> str:
    """Determine which column holds the actual stat value."""
    for col in ("actual", "actual_value"):
        if col in df.columns:
            return col
    return "actual"


def get_market_column(df: pd.DataFrame) -> str:
    """Determine which column holds the market type."""
    if "market" in df.columns:
        return "market"
    return "market"


def get_bet_type_column(df: pd.DataFrame) -> str:
    """Determine which column holds the bet type/side."""
    for col in ("bet_type", "side"):
        if col in df.columns:
            return col
    return "bet_type"


def fetch_match_results(
    client: FootballAPIClient,
    fixture_ids: List[int],
    verbose: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Fetch match results from API-Football for a list of fixture IDs.

    Uses rate limiting to stay within API limits. Caches results so each
    fixture is fetched only once.

    Returns dict mapping fixture_id -> parsed result dict.
    """
    results: Dict[int, Dict[str, Any]] = {}
    total = len(fixture_ids)

    logger.info(f"Fetching results for {total} fixtures from API-Football...")

    for i, fix_id in enumerate(fixture_ids):
        if verbose:
            logger.info(f"  [{i + 1}/{total}] Fetching fixture {fix_id}...")

        try:
            # Use the raw _make_request to get full fixture data with statistics
            response = client._make_request(
                "/fixtures",
                {"id": fix_id},
            )

            parsed = parse_fixture_response(response)
            if parsed is not None:
                results[fix_id] = parsed
                if verbose:
                    logger.info(
                        f"    -> {parsed['home_goals']}-{parsed['away_goals']} "
                        f"(corners={parsed.get('total_corners', 'N/A')}, "
                        f"fouls={parsed.get('total_fouls', 'N/A')}, "
                        f"shots={parsed.get('total_shots', 'N/A')}, "
                        f"cards={parsed.get('total_cards', 'N/A')})"
                    )
            else:
                if verbose:
                    logger.info(f"    -> Not finished or no data")

        except Exception as e:
            logger.warning(f"  [{i + 1}/{total}] Error fetching fixture {fix_id}: {e}")
            if "Daily limit" in str(e):
                logger.error("API daily limit reached. Stopping.")
                break

        # Rate limit: sleep between calls (the TokenBucket in the client handles
        # per-minute rate, but we add a safety sleep to be explicit)
        if i < total - 1:
            time.sleep(CALL_INTERVAL_SECONDS)

    logger.info(f"Fetched results for {len(results)}/{total} fixtures")
    return results


def needs_statistics(market: str) -> bool:
    """Check if a market requires match statistics (not just goals)."""
    market_upper = str(market).upper().replace(" ", "_")
    return market_upper in ("SHOTS", "FOULS", "CORNERS", "CARDS", "FOULS_OVER", "FOULS_UNDER")


def update_file(
    csv_path: Path,
    df: pd.DataFrame,
    match_results: Dict[int, Dict[str, Any]],
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Update a single recommendation CSV file with match results.

    Returns summary dict with counts of updates.
    """
    date_col = get_date_column(df)
    result_col = get_result_column(df)
    actual_col = get_actual_column(df)
    market_col = get_market_column(df)
    bet_type_col = get_bet_type_column(df)

    # Ensure result and actual columns exist
    if result_col not in df.columns:
        df[result_col] = ""
    if actual_col not in df.columns:
        df[actual_col] = ""

    updates = 0
    skipped_no_fixture = 0
    skipped_future = 0
    skipped_no_result = 0
    skipped_no_stats = 0
    already_settled = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Skip already settled bets
        if not is_pending(row.get(result_col)):
            current_result = str(row.get(result_col, "")).strip().upper()
            if current_result in ("W", "L", "WON", "LOST", "PUSH", "VOID"):
                already_settled += 1
                continue

        # Check fixture_id
        fixture_id_raw = row.get("fixture_id")
        if pd.isna(fixture_id_raw) or str(fixture_id_raw).strip() in ("", "nan", "None"):
            skipped_no_fixture += 1
            continue

        try:
            fixture_id = int(float(str(fixture_id_raw).strip()))
        except (ValueError, TypeError):
            skipped_no_fixture += 1
            continue

        # Check if match date is in the past
        if date_col and date_col in df.columns:
            date_val = row.get(date_col, "")
            if not is_past_date(date_val):
                skipped_future += 1
                continue

        # Look up match result
        if fixture_id not in match_results:
            skipped_no_result += 1
            continue

        match_result = match_results[fixture_id]

        # Get market and bet_type
        market = str(row.get(market_col, "")).strip()
        bet_type = str(row.get(bet_type_col, "")).strip()

        # Handle cases where bet_type is same as market (e.g., OVER_2.5 / OVER_2.5)
        if not bet_type or bet_type.lower() in ("nan", "none", ""):
            bet_type = market

        # Get line value
        line_raw = row.get("line", 0)
        try:
            line = float(str(line_raw).strip()) if not pd.isna(line_raw) and str(line_raw).strip() not in ("", "nan") else 0.0
        except (ValueError, TypeError):
            line = 0.0

        # Check if we need statistics but don't have them
        if needs_statistics(market) and "total_shots" not in match_result:
            skipped_no_stats += 1
            continue

        # Determine outcome
        outcome, actual_value = determine_outcome(market, bet_type, line, match_result)
        if outcome is None:
            if verbose:
                logger.debug(
                    f"  Could not determine outcome for row {idx}: "
                    f"market={market}, bet_type={bet_type}, line={line}"
                )
            continue

        # Map to the file's result convention
        # Some files use W/L, others use WON/LOST
        existing_results = df[result_col].dropna().unique()
        use_long_form = any(
            str(r).upper() in ("WON", "LOST") for r in existing_results
        )
        if use_long_form:
            result_str = "WON" if outcome == "W" else "LOST"
        else:
            result_str = outcome

        # Calculate PnL if odds are available
        odds_raw = row.get("odds")
        pnl = None
        if odds_raw is not None and str(odds_raw).strip() not in ("", "nan", "None", "0", "0.0"):
            try:
                odds_val = float(str(odds_raw).strip())
                if odds_val > 0:
                    pnl = (odds_val - 1.0) if outcome == "W" else -1.0
            except (ValueError, TypeError):
                pass

        if verbose:
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            logger.info(
                f"  {home} vs {away} | {market} {bet_type} {line} | "
                f"{result_str} (actual={actual_value}) | pnl={pnl}"
            )

        # Apply updates
        df.iat[idx, df.columns.get_loc(result_col)] = result_str
        if actual_value is not None:
            df.iat[idx, df.columns.get_loc(actual_col)] = str(actual_value)

        # Update PnL column if it exists
        if "pnl" in df.columns and pnl is not None:
            df.iat[idx, df.columns.get_loc("pnl")] = str(pnl)

        # Update settled_at if it exists
        if "settled_at" in df.columns:
            df.iat[idx, df.columns.get_loc("settled_at")] = datetime.now().isoformat()

        # Update status column if separate from result
        if "status" in df.columns and result_col != "status":
            df.iat[idx, df.columns.get_loc("status")] = result_str

        updates += 1

    # Save updated file
    if updates > 0 and not dry_run:
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved {updates} updates to {csv_path.name}")
    elif updates > 0:
        logger.info(f"  [DRY RUN] Would save {updates} updates to {csv_path.name}")

    return {
        "file": csv_path.name,
        "updates": updates,
        "already_settled": already_settled,
        "skipped_no_fixture": skipped_no_fixture,
        "skipped_future": skipped_future,
        "skipped_no_result": skipped_no_result,
        "skipped_no_stats": skipped_no_stats,
    }


def collect_pending_fixture_ids(
    files: List[Tuple[Path, pd.DataFrame]],
    date_range: Optional[Tuple[str, str]] = None,
) -> List[int]:
    """Collect unique fixture IDs for all pending bets across all files."""
    fixture_ids: set[int] = set()

    for csv_path, df in files:
        date_col = get_date_column(df)
        result_col = get_result_column(df)

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Skip settled bets
            if not is_pending(row.get(result_col)):
                current_result = str(row.get(result_col, "")).strip().upper()
                if current_result in ("W", "L", "WON", "LOST", "PUSH", "VOID"):
                    continue

            # Check fixture_id exists
            fixture_id_raw = row.get("fixture_id")
            if pd.isna(fixture_id_raw) or str(fixture_id_raw).strip() in ("", "nan", "None"):
                continue

            try:
                fixture_id = int(float(str(fixture_id_raw).strip()))
            except (ValueError, TypeError):
                continue

            # Check if match date is in the past
            if date_col and date_col in df.columns:
                date_val = row.get(date_col, "")
                if not is_past_date(date_val):
                    continue

            fixture_ids.add(fixture_id)

    return sorted(fixture_ids)


def print_summary_report(
    files: List[Tuple[Path, pd.DataFrame]],
    file_summaries: List[Dict[str, Any]],
) -> None:
    """Print summary report of all settled and pending bets."""
    print()
    print("=" * 70)
    print("RESULTS UPDATE SUMMARY")
    print("=" * 70)

    # Aggregate file update stats
    total_updates = sum(s["updates"] for s in file_summaries)
    total_no_fixture = sum(s["skipped_no_fixture"] for s in file_summaries)
    total_future = sum(s["skipped_future"] for s in file_summaries)
    total_no_result = sum(s["skipped_no_result"] for s in file_summaries)
    total_no_stats = sum(s["skipped_no_stats"] for s in file_summaries)

    print(f"\nUpdates this run: {total_updates}")
    print(f"Skipped (no fixture_id): {total_no_fixture}")
    print(f"Skipped (future match): {total_future}")
    print(f"Skipped (result not available): {total_no_result}")
    print(f"Skipped (stats not available): {total_no_stats}")

    # Collect all rows from updated files for per-market summary
    all_rows: List[Dict[str, Any]] = []
    for csv_path, df in files:
        result_col = get_result_column(df)
        actual_col = get_actual_column(df)
        market_col = get_market_column(df)

        for idx in range(len(df)):
            row = df.iloc[idx]
            result_val = str(row.get(result_col, "")).strip().upper()
            market_val = str(row.get(market_col, "")).strip().upper()

            # Normalize result values
            if result_val in ("W", "WON"):
                result_val = "W"
            elif result_val in ("L", "LOST"):
                result_val = "L"
            else:
                result_val = "PENDING"

            odds_raw = row.get("odds")
            odds_val = 0.0
            if odds_raw is not None:
                try:
                    odds_val = float(str(odds_raw).strip())
                except (ValueError, TypeError):
                    pass

            all_rows.append({
                "market": market_val,
                "result": result_val,
                "odds": odds_val,
            })

    if not all_rows:
        print("\nNo recommendation data found.")
        return

    rows_df = pd.DataFrame(all_rows)

    # Per-market summary
    print("\n" + "-" * 70)
    print(f"{'Market':<18} {'Settled':>8} {'Wins':>6} {'Losses':>7} {'Win Rate':>9} {'ROI':>8} {'Pending':>8}")
    print("-" * 70)

    markets = sorted(rows_df["market"].unique())
    total_settled = 0
    total_wins = 0
    total_losses = 0
    total_profit = 0.0
    total_staked = 0
    total_pending = 0

    for market in markets:
        market_df = rows_df[rows_df["market"] == market]
        settled = market_df[market_df["result"].isin(["W", "L"])]
        pending = market_df[market_df["result"] == "PENDING"]

        n_settled = len(settled)
        n_wins = (settled["result"] == "W").sum()
        n_losses = (settled["result"] == "L").sum()
        n_pending = len(pending)

        win_rate = (n_wins / n_settled * 100) if n_settled > 0 else 0.0

        # Calculate ROI using actual odds where available
        profit = 0.0
        staked = 0
        for _, r in settled.iterrows():
            if r["odds"] > 0:
                staked += 1
                if r["result"] == "W":
                    profit += r["odds"] - 1.0
                else:
                    profit -= 1.0

        roi = (profit / staked * 100) if staked > 0 else 0.0

        total_settled += n_settled
        total_wins += n_wins
        total_losses += n_losses
        total_profit += profit
        total_staked += staked
        total_pending += n_pending

        roi_str = f"{roi:+.1f}%" if staked > 0 else "N/A"
        print(
            f"{market:<18} {n_settled:>8} {n_wins:>6} {n_losses:>7} "
            f"{win_rate:>8.1f}% {roi_str:>8} {n_pending:>8}"
        )

    # Overall totals
    print("-" * 70)
    overall_win_rate = (total_wins / total_settled * 100) if total_settled > 0 else 0.0
    overall_roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    roi_str = f"{overall_roi:+.1f}%" if total_staked > 0 else "N/A"
    print(
        f"{'TOTAL':<18} {total_settled:>8} {total_wins:>6} {total_losses:>7} "
        f"{overall_win_rate:>8.1f}% {roi_str:>8} {total_pending:>8}"
    )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update match results for pending betting recommendations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview updates without saving to files",
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Only process bets within date range (YYYY-MM-DD YYYY-MM-DD)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging with per-bet details",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Base directory for recommendations (default: {RECOMMENDATIONS_DIR})",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    base_dir = Path(args.dir) if args.dir else RECOMMENDATIONS_DIR
    if not base_dir.exists():
        logger.error(f"Recommendations directory not found: {base_dir}")
        sys.exit(1)

    date_range = tuple(args.date_range) if args.date_range else None

    # Step 1: Load all recommendation files
    files = load_recommendation_files(base_dir, date_range=date_range)
    if not files:
        logger.info("No recommendation files found. Nothing to do.")
        return

    # Step 2: Collect all pending fixture IDs
    pending_fixture_ids = collect_pending_fixture_ids(files, date_range=date_range)
    logger.info(f"Found {len(pending_fixture_ids)} unique fixtures with pending bets")

    if not pending_fixture_ids:
        logger.info("No pending bets to update.")
        print_summary_report(files, [])
        return

    # Step 3: Fetch results from API-Football
    if args.dry_run:
        logger.info("[DRY RUN] Fetching results (will not save updates)...")

    client = FootballAPIClient()
    match_results = fetch_match_results(
        client, pending_fixture_ids, verbose=args.verbose
    )

    if not match_results:
        logger.info("No match results available (matches may not be finished yet).")
        print_summary_report(files, [])
        return

    # Step 4: Update each file
    file_summaries: List[Dict[str, Any]] = []
    for csv_path, df in files:
        summary = update_file(
            csv_path, df, match_results, dry_run=args.dry_run, verbose=args.verbose
        )
        file_summaries.append(summary)

    # Step 5: Print summary report (using the now-updated dataframes)
    print_summary_report(files, file_summaries)


if __name__ == "__main__":
    main()
