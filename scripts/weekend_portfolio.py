"""Weekend paper-trading portfolio: select picks, generate reports, send Telegram."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEEKEND_DIR = PROJECT_ROOT / "data" / "weekend"
REPORTS_DIR = WEEKEND_DIR / "reports"
REC_DIR = PROJECT_ROOT / "data" / "05-recommendations"


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> int:
    """Select a constrained paper portfolio from today's recommendations."""
    rec_file = Path(args.rec_file)
    if not rec_file.exists():
        logger.error(f"Rec file not found: {rec_file}")
        return 1

    with open(rec_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("Rec file is empty")
        return 0

    # Filter: positive edge >= min_edge, not yet settled
    candidates = []
    for r in rows:
        try:
            edge = float(r.get("edge", 0))
        except (ValueError, TypeError):
            continue
        result = (r.get("result") or "").strip()
        if edge >= args.min_edge and result not in ("W", "L", "P"):
            candidates.append(r)

    candidates.sort(key=lambda r: float(r.get("edge", 0)), reverse=True)

    # Greedy selection with constraints
    selected: list[dict] = []
    market_counts: dict[str, int] = {}
    match_seen: set[str] = set()

    for r in candidates:
        if len(selected) >= args.max_bets:
            break
        market = r.get("market", r.get("bet_type", "unknown"))
        fixture_id = r.get("fixture_id", "")
        match_key = fixture_id or f"{r.get('home_team', '')}_{r.get('away_team', '')}"

        if market_counts.get(market, 0) >= args.max_per_market:
            continue
        if match_key in match_seen:
            continue

        selected.append(r)
        market_counts[market] = market_counts.get(market, 0) + 1
        match_seen.add(match_key)

    if not selected:
        logger.info("No picks qualify after constraints")
        return 0

    # Write picks CSV
    WEEKEND_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = WEEKEND_DIR / f"weekend_picks_{date_str}.csv"

    fieldnames = [
        "date",
        "fixture_id",
        "home_team",
        "away_team",
        "league",
        "market",
        "bet_type",
        "line",
        "probability",
        "edge",
        "odds",
        "stake",
        "model",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in selected:
            writer.writerow(
                {
                    "date": r.get("date", date_str),
                    "fixture_id": r.get("fixture_id", ""),
                    "home_team": r.get("home_team", ""),
                    "away_team": r.get("away_team", ""),
                    "league": r.get("league", ""),
                    "market": r.get("market", ""),
                    "bet_type": r.get("bet_type", ""),
                    "line": r.get("line", ""),
                    "probability": r.get("probability", ""),
                    "edge": r.get("edge", ""),
                    "odds": r.get("odds", ""),
                    "stake": args.stake,
                    "model": r.get("edge_source", ""),
                }
            )

    logger.info(
        f"Selected {len(selected)} picks (from {len(candidates)} candidates) → {out_path}"
    )
    per_mkt = ", ".join(f"{k}={v}" for k, v in sorted(market_counts.items()))
    logger.info(f"  Markets: {per_mkt}")
    logger.info(f"  Total stake: {len(selected) * args.stake} PLN")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _last_weekend_dates(ref_date: datetime | None = None) -> list[datetime]:
    """Return Fri/Sat/Sun dates for the most recent weekend before ref_date.

    If ref_date is Monday, returns the immediately preceding Fri-Sun.
    """
    d = ref_date or datetime.now()
    # Walk back to most recent Sunday (if not already Sun)
    while d.weekday() != 6:  # 6 = Sunday
        d -= timedelta(days=1)
    sun = d
    sat = sun - timedelta(days=1)
    fri = sun - timedelta(days=2)
    return [fri, sat, sun]


def cmd_report(args: argparse.Namespace) -> int:
    """Generate a weekend P&L report from picks + settled rec CSVs."""
    if args.weekend:
        ref = datetime.strptime(args.weekend, "%Y-%m-%d")
    else:
        ref = datetime.now()

    weekend_dates = _last_weekend_dates(ref)
    date_strs = [d.strftime("%Y%m%d") for d in weekend_dates]
    day_labels = ["Fri", "Sat", "Sun"]

    logger.info(
        f"Report for weekend: {weekend_dates[0].strftime('%b %d')} – "
        f"{weekend_dates[2].strftime('%b %d, %Y')}"
    )

    # Load picks for each day
    all_picks: list[dict] = []
    picks_by_day: dict[str, list[dict]] = {}
    for ds, label in zip(date_strs, day_labels):
        picks_file = WEEKEND_DIR / f"weekend_picks_{ds}.csv"
        if picks_file.exists():
            with open(picks_file) as f:
                day_picks = list(csv.DictReader(f))
            all_picks.extend(day_picks)
            picks_by_day[label] = day_picks
            logger.info(f"  {label} ({ds}): {len(day_picks)} picks")
        else:
            picks_by_day[label] = []
            logger.info(f"  {label} ({ds}): no picks file")

    if not all_picks:
        logger.warning("No picks found for this weekend")
        return 0

    # Load rec CSVs for matching results
    rec_results: dict[str, dict] = {}  # (fixture_id, market) → rec row
    for ds in date_strs:
        for rec_file in sorted(REC_DIR.glob(f"rec_{ds}_*.csv")):
            if "_estimated" in rec_file.name:
                continue
            with open(rec_file) as f:
                for row in csv.DictReader(f):
                    fid = row.get("fixture_id", "")
                    market = row.get("market", "")
                    if fid and market:
                        key = f"{fid}_{market}"
                        rec_results[key] = row

    # Match picks to results
    total_staked = 0.0
    total_pnl = 0.0
    wins = 0
    losses = 0
    pushes = 0
    unsettled = 0
    market_stats: dict[str, dict] = {}
    day_stats: dict[str, dict] = {
        label: {"bets": 0, "pnl": 0.0, "staked": 0.0}
        for label in day_labels
    }

    settled_picks: list[dict] = []

    for label, day_picks in picks_by_day.items():
        for pick in day_picks:
            fid = pick.get("fixture_id", "")
            market = pick.get("market", "")
            key = f"{fid}_{market}"
            try:
                stake = float(pick.get("stake", 0))
            except (ValueError, TypeError):
                stake = 0.0
            try:
                odds = float(pick.get("odds", 0))
            except (ValueError, TypeError):
                odds = 0.0

            rec = rec_results.get(key, {})
            result = (rec.get("result") or pick.get("result", "")).strip()

            total_staked += stake
            day_stats[label]["bets"] += 1
            day_stats[label]["staked"] += stake

            if market not in market_stats:
                market_stats[market] = {"wins": 0, "losses": 0, "pushes": 0, "pnl": 0.0}

            if result == "W":
                profit = stake * (odds - 1) if odds > 1 else 0
                total_pnl += profit
                wins += 1
                market_stats[market]["wins"] += 1
                market_stats[market]["pnl"] += profit
                day_stats[label]["pnl"] += profit
            elif result == "L":
                total_pnl -= stake
                losses += 1
                market_stats[market]["losses"] += 1
                market_stats[market]["pnl"] -= stake
                day_stats[label]["pnl"] -= stake
            elif result == "P":
                pushes += 1
                market_stats[market]["pushes"] += 1
            else:
                unsettled += 1

            settled_picks.append(
                {**pick, "result": result, "pnl": total_pnl}
            )

    total_bets = wins + losses + pushes + unsettled
    settled_bets = wins + losses + pushes
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
    win_rate = (wins / settled_bets * 100) if settled_bets > 0 else 0

    # Build report
    iso_week = weekend_dates[2].isocalendar()
    week_label = f"{iso_week[0]}-W{iso_week[1]:02d}"

    report = {
        "week": week_label,
        "dates": {
            "friday": weekend_dates[0].strftime("%Y-%m-%d"),
            "saturday": weekend_dates[1].strftime("%Y-%m-%d"),
            "sunday": weekend_dates[2].strftime("%Y-%m-%d"),
        },
        "summary": {
            "total_bets": total_bets,
            "settled": settled_bets,
            "unsettled": unsettled,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(win_rate, 1),
            "staked": round(total_staked, 2),
            "pnl": round(total_pnl, 2),
            "roi": round(roi, 1),
        },
        "per_market": {
            m: {
                "wins": s["wins"],
                "losses": s["losses"],
                "pushes": s["pushes"],
                "pnl": round(s["pnl"], 2),
            }
            for m, s in sorted(market_stats.items())
        },
        "per_day": {
            label: {
                "bets": s["bets"],
                "staked": round(s["staked"], 2),
                "pnl": round(s["pnl"], 2),
            }
            for label, s in day_stats.items()
        },
    }

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"weekend_report_{week_label}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved: {report_path}")
    s = report["summary"]
    logger.info(
        f"  Bets: {s['total_bets']} | W: {s['wins']} L: {s['losses']} "
        f"| Win: {s['win_rate']}%"
    )
    logger.info(
        f"  Staked: {s['staked']} PLN | P&L: {s['pnl']:+.2f} PLN "
        f"| ROI: {s['roi']:+.1f}%"
    )
    if unsettled > 0:
        logger.warning(f"  {unsettled} bets still unsettled!")

    print(report_path)  # stdout for downstream scripts
    return 0


# ---------------------------------------------------------------------------
# telegram
# ---------------------------------------------------------------------------


def cmd_telegram(args: argparse.Namespace) -> int:
    """Send weekend report summary to Telegram."""
    report_path = Path(args.report)
    if not report_path.exists():
        logger.error(f"Report not found: {report_path}")
        return 1

    with open(report_path) as f:
        report = json.load(f)

    from src.notifications import TelegramNotifier, format_weekly_report

    parts = format_weekly_report(weekend=report)
    notifier = TelegramNotifier()

    if not notifier.is_configured:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        return 1

    ok = notifier.send_parts(parts)
    if ok:
        logger.info("Telegram report sent!")
    else:
        logger.error("Telegram send failed")
        return 1

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Weekend paper-trading portfolio manager"
    )
    sub = parser.add_subparsers(dest="command")

    # select
    p_sel = sub.add_parser("select", help="Select paper portfolio from rec CSV")
    p_sel.add_argument("--rec-file", required=True, help="Path to rec CSV")
    p_sel.add_argument(
        "--max-bets", type=int, default=10, help="Max bets per day (default 10)"
    )
    p_sel.add_argument(
        "--max-per-market",
        type=int,
        default=5,
        help="Max bets per market (default 5)",
    )
    p_sel.add_argument(
        "--min-edge",
        type=float,
        default=5.0,
        help="Minimum edge %% to include (default 5.0)",
    )
    p_sel.add_argument(
        "--stake",
        type=float,
        default=100.0,
        help="Flat stake per bet in PLN (default 100)",
    )

    # report
    p_rep = sub.add_parser("report", help="Generate weekend P&L report")
    p_rep.add_argument(
        "--weekend",
        type=str,
        default=None,
        help="Reference date YYYY-MM-DD (defaults to most recent weekend)",
    )

    # telegram
    p_tg = sub.add_parser("telegram", help="Send report to Telegram")
    p_tg.add_argument("--report", required=True, help="Path to report JSON")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "select": cmd_select,
        "report": cmd_report,
        "telegram": cmd_telegram,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
