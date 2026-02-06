#!/usr/bin/env python3
"""
Pre-Kickoff Re-Prediction with Lineup Data.

Standalone script that re-runs predictions for matches with confirmed lineups.
Does NOT modify morning predictions — outputs to a separate CSV.

Flow:
1. Reads morning recommendation CSV(s)
2. For matches kicking off in 15-70 min, fetches lineups from API-Football
3. Saves lineups to data/06-prematch/{fixture_id}/
4. Re-runs generate_daily_recommendations.py for those fixtures
5. Computes delta: updated_prob - morning_prob
6. Outputs data/05-recommendations/pre_kickoff_{date}.csv
7. Sends Telegram notification for UPGRADE/DOWNGRADE actions

Usage:
    python scripts/pre_kickoff_repredict.py
    python scripts/pre_kickoff_repredict.py --delta-threshold 0.03
    python scripts/pre_kickoff_repredict.py --api-budget 10
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _find_morning_recs(date_str: str) -> List[Path]:
    """Find morning recommendation CSV(s) for today."""
    rec_dir = project_root / "data" / "05-recommendations"
    return sorted(rec_dir.glob(f"rec_{date_str}_*.csv"))


def _load_schedule() -> List[Dict]:
    """Load today's schedule."""
    schedule_file = project_root / "data" / "06-prematch" / "today_schedule.json"
    if not schedule_file.exists():
        return []
    with open(schedule_file) as f:
        return json.load(f).get("matches", [])


def _matches_in_window(
    matches: List[Dict],
    window_start_min: int = 15,
    window_end_min: int = 70,
) -> List[Dict]:
    """Filter to matches kicking off in the specified window."""
    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=window_start_min)
    end = now + timedelta(minutes=window_end_min)

    result = []
    for m in matches:
        kickoff_str = m.get("kickoff", "")
        if not kickoff_str:
            continue
        try:
            kickoff = datetime.fromisoformat(str(kickoff_str))
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        if start <= kickoff <= end:
            m["mins_until"] = int((kickoff - now).total_seconds() / 60)
            result.append(m)

    return result


def fetch_lineups_from_api(
    fixture_ids: List[int],
    api_budget: int = 20,
) -> Dict[int, Dict]:
    """
    Fetch lineups from API-Football for given fixtures.

    Args:
        fixture_ids: List of fixture IDs to fetch lineups for.
        api_budget: Maximum API calls to make.

    Returns:
        Dict mapping fixture_id -> lineup data.
    """
    results = {}
    calls_made = 0

    try:
        from src.data_collection.api_client import FootballAPIClient
        client = FootballAPIClient()
    except Exception as e:
        logger.error(f"Cannot initialize API client: {e}")
        return results

    for fid in fixture_ids[:api_budget]:
        if calls_made >= api_budget:
            logger.info(f"API budget exhausted ({calls_made}/{api_budget})")
            break

        try:
            response = client._make_request('/fixtures/lineups', {"fixture": fid})
            calls_made += 1

            if response and response.get("results", 0) > 0:
                raw = response.get("response", [])
                if len(raw) >= 2:
                    results[fid] = {
                        "home": _parse_api_lineup(raw[0]),
                        "away": _parse_api_lineup(raw[1]),
                    }
                    logger.info(f"  Fetched lineup for fixture {fid}")
                else:
                    logger.info(f"  Fixture {fid}: incomplete lineup data ({len(raw)} teams)")
            else:
                logger.info(f"  Fixture {fid}: no lineup available yet")

        except Exception as e:
            logger.warning(f"  Failed to fetch lineup for fixture {fid}: {e}")
            calls_made += 1

    logger.info(f"Fetched lineups for {len(results)}/{len(fixture_ids)} fixtures ({calls_made} API calls)")
    return results


def _parse_api_lineup(team_data: Dict) -> Optional[Dict]:
    """Parse API-Football lineup response into our format."""
    if not team_data:
        return None

    start_xi = team_data.get("startXI", [])
    players = []
    for entry in start_xi:
        player = entry.get("player", {})
        if player.get("id"):
            players.append({
                "id": player["id"],
                "name": player.get("name", ""),
            })

    if not players:
        return None

    return {
        "starting_xi": players,
        "formation": team_data.get("formation", ""),
    }


def save_lineup_to_cache(fixture_id: int, lineup_data: Dict) -> None:
    """Save fetched lineup to prematch cache for generate_daily_recommendations to pick up."""
    cache_dir = project_root / "data" / "06-prematch" / str(fixture_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = cache_dir / "lineup_window_latest.json"
    with open(out_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "lineups": {
                "available": True,
                "home": lineup_data.get("home"),
                "away": lineup_data.get("away"),
            },
        }, f, indent=2)

    logger.info(f"Saved lineup cache: {out_path}")


def compute_deltas(
    morning_df: pd.DataFrame,
    updated_df: pd.DataFrame,
    delta_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Compute probability deltas between morning and pre-kickoff predictions.

    Args:
        morning_df: Morning recommendations DataFrame.
        updated_df: Updated recommendations DataFrame.
        delta_threshold: Minimum absolute delta to report.

    Returns:
        DataFrame with delta information.
    """
    if morning_df.empty or updated_df.empty:
        return pd.DataFrame()

    # Merge on match + market
    merge_keys = ["fixture_id", "market", "bet_type"]
    available_keys = [k for k in merge_keys if k in morning_df.columns and k in updated_df.columns]
    if not available_keys:
        return pd.DataFrame()

    merged = morning_df.merge(
        updated_df[available_keys + ["probability"]],
        on=available_keys,
        how="inner",
        suffixes=("_morning", "_updated"),
    )

    if merged.empty:
        return pd.DataFrame()

    merged["delta"] = merged["probability_updated"] - merged["probability_morning"]
    merged["abs_delta"] = merged["delta"].abs()

    # Filter by threshold
    significant = merged[merged["abs_delta"] >= delta_threshold].copy()
    significant["action"] = significant["delta"].apply(
        lambda d: "UPGRADE" if d > 0 else "DOWNGRADE"
    )

    return significant.sort_values("abs_delta", ascending=False)


def send_telegram_notification(deltas_df: pd.DataFrame) -> None:
    """Send Telegram notification for significant prediction changes."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info("Telegram credentials not set, skipping notification")
        return

    if deltas_df.empty:
        return

    try:
        import requests
    except ImportError:
        logger.warning("requests not available, skipping Telegram")
        return

    sep = "━━━━━━━━━━━━━━━━━━━━━"
    lines = ["🔄 <b>PRE-KICKOFF UPDATE</b>", sep, ""]

    for _, row in deltas_df.head(10).iterrows():
        action = row.get("action", "")
        emoji = "⬆️" if action == "UPGRADE" else "⬇️"
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        market = row.get("market", "?")
        delta = row.get("delta", 0) * 100
        new_prob = row.get("probability_updated", 0)

        lines.append(f"{emoji} <b>{home} vs {away}</b>")
        lines.append(f"   {market}: {delta:+.1f}pp → {new_prob:.1%}")
        lines.append("")

    lines.append(sep)
    message = "\n".join(lines)

    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
    )
    logger.info(f"Telegram notification sent ({len(deltas_df)} updates)")


def repredict_with_lineups(
    delta_threshold: float = 0.02,
    api_budget: int = 20,
) -> pd.DataFrame:
    """
    Main re-prediction flow.

    Returns:
        DataFrame with pre-kickoff predictions (or empty if nothing to update).
    """
    today = datetime.now().strftime("%Y%m%d")

    # 1. Find morning predictions
    morning_files = _find_morning_recs(today)
    if not morning_files:
        logger.info("No morning predictions found for today")
        return pd.DataFrame()

    morning_df = pd.concat([pd.read_csv(f) for f in morning_files], ignore_index=True)
    logger.info(f"Loaded {len(morning_df)} morning predictions from {len(morning_files)} file(s)")

    # 2. Find matches in pre-kickoff window
    schedule = _load_schedule()
    in_window = _matches_in_window(schedule)
    if not in_window:
        logger.info("No matches in pre-kickoff window (15-70 min)")
        return pd.DataFrame()

    logger.info(f"Found {len(in_window)} matches in pre-kickoff window:")
    for m in in_window:
        logger.info(f"  {m['home_team']} vs {m['away_team']} in {m['mins_until']} mins")

    # 3. Fetch lineups from API-Football
    fixture_ids = [m["fixture_id"] for m in in_window if m.get("fixture_id")]
    lineups = fetch_lineups_from_api(fixture_ids, api_budget=api_budget)

    if not lineups:
        logger.info("No lineups available for matches in window")
        return pd.DataFrame()

    # 4. Save lineups to prematch cache
    for fid, lineup_data in lineups.items():
        save_lineup_to_cache(fid, lineup_data)

    # 5. Create schedule file for just these fixtures
    matches_with_lineups = [m for m in in_window if m.get("fixture_id") in lineups]
    temp_schedule = project_root / "data" / "06-prematch" / "pre_kickoff_schedule.json"
    with open(temp_schedule, "w") as f:
        json.dump({"matches": matches_with_lineups, "total_matches": len(matches_with_lineups)}, f)

    # 6. Re-run predictions via generate_daily_recommendations
    logger.info(f"Re-running predictions for {len(matches_with_lineups)} matches with lineups...")
    from experiments.generate_daily_recommendations import generate_sniper_predictions

    updated_predictions = generate_sniper_predictions(
        matches_with_lineups,
        min_edge_pct=5.0,
    )

    if not updated_predictions:
        logger.info("No updated predictions generated")
        return pd.DataFrame()

    updated_df = pd.DataFrame(updated_predictions)

    # 7. Compute deltas
    deltas = compute_deltas(morning_df, updated_df, delta_threshold)

    # 8. Save pre-kickoff predictions
    output_dir = project_root / "data" / "05-recommendations"
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"pre_kickoff_{date_str}.csv"
    updated_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(updated_df)} pre-kickoff predictions to {output_path}")

    # 9. Send notification
    if not deltas.empty:
        logger.info(f"\n{len(deltas)} significant changes (delta >= {delta_threshold*100:.0f}pp):")
        for _, row in deltas.iterrows():
            logger.info(
                f"  {row.get('action', '')} {row.get('home_team', '')} vs {row.get('away_team', '')} "
                f"{row.get('market', '')}: {row.get('delta', 0)*100:+.1f}pp"
            )
        send_telegram_notification(deltas)
    else:
        logger.info("No significant probability changes")

    return updated_df


def main():
    parser = argparse.ArgumentParser(description="Pre-kickoff re-prediction with lineup data")
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.02,
        help="Minimum probability delta to report (default: 0.02 = 2pp)",
    )
    parser.add_argument(
        "--api-budget",
        type=int,
        default=20,
        help="Maximum API calls for lineup fetching (default: 20)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PRE-KICKOFF RE-PREDICTION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Delta threshold: {args.delta_threshold*100:.0f}pp")
    print(f"API budget: {args.api_budget} calls")
    print("=" * 60)

    result = repredict_with_lineups(
        delta_threshold=args.delta_threshold,
        api_budget=args.api_budget,
    )

    if result.empty:
        print("\nNo pre-kickoff updates")
        return 0

    print(f"\nGenerated {len(result)} updated predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
