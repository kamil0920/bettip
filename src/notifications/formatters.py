"""Pure formatting functions for Telegram notification messages.

Each function returns a list[str] of message parts (each <= 4096 chars).
No I/O — only string formatting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.notifications.market_utils import classify_recommendations, is_real_odds_market

SEP = "\u2501" * 21  # ━━━━━━━━━━━━━━━━━━━━━

NUMBER_EMOJIS = [
    "1\ufe0f\u20e3",
    "2\ufe0f\u20e3",
    "3\ufe0f\u20e3",
    "4\ufe0f\u20e3",
    "5\ufe0f\u20e3",
    "6\ufe0f\u20e3",
    "7\ufe0f\u20e3",
    "8\ufe0f\u20e3",
    "9\ufe0f\u20e3",
    "\U0001f51f",
]


# ---------------------------------------------------------------------------
# 1. Match Day Digest  (Fri/Sat/Sun 7:00)
# ---------------------------------------------------------------------------


def format_match_day_digest(
    recs: list[dict[str, Any]],
    drift_signals: list[dict[str, Any]] | None = None,
    total_matches: int = 0,
) -> list[str]:
    """Format morning predictions as a market-aware digest.

    Real-odds picks show edge + probability.
    Model-only picks show probability only — never edge/ROI/odds.

    Args:
        recs: Recommendation dicts (from CSV rows).
        drift_signals: Optional list of ``{"market": ..., "signal": ...}`` dicts.
        total_matches: Total scheduled matches for the day.

    Returns:
        List of message parts (each <= 4096 chars).
    """
    if not recs:
        return [f"\u26bd <b>MATCH DAY DIGEST \u2022 {_today()}</b>\n{SEP}\nNo picks today."]

    real_odds, model_only = classify_recommendations(recs)

    lines: list[str] = [
        f"\u26bd <b>MATCH DAY DIGEST \u2022 {_today()}</b>",
        SEP,
        f"\U0001f4ca {len(recs)} picks from {total_matches} matches",
        "",
    ]

    # --- Real-odds section (sorted by edge desc) ---
    if real_odds:
        lines.append("<b>\U0001f4b0 REAL ODDS</b>")
        for i, rec in enumerate(real_odds[:10]):
            num = NUMBER_EMOJIS[i] if i < len(NUMBER_EMOJIS) else f"{i + 1}."
            match_name = _match_name(rec)
            market = rec.get("market", rec.get("bet_type", ""))
            prob = _pct(rec.get("probability", 0))
            edge = float(rec.get("edge", 0))
            emoji = "\U0001f525" if edge >= 15 else "\u26a1"

            lines.append(f"{num} {emoji} <b>{match_name}</b>")
            lines.append(f"   {market} | Prob {prob}% | Edge +{edge:.0f}%")
        if len(real_odds) > 10:
            lines.append(f"   ...+{len(real_odds) - 10} more")
        lines.append("")

    # --- Model-only section (sorted by probability desc) ---
    if model_only:
        lines.append("<b>\U0001f916 MODEL ONLY</b> <i>(check real odds before betting)</i>")
        for i, rec in enumerate(model_only[:5]):
            match_name = _match_name(rec)
            market = rec.get("market", rec.get("bet_type", ""))
            prob = _pct(rec.get("probability", 0))

            lines.append(f"  \u2022 <b>{match_name}</b>")
            lines.append(f"    {market} | Prob {prob}%")
        if len(model_only) > 5:
            lines.append(f"   ...+{len(model_only) - 5} more")
        lines.append("")

    # --- Drift alerts ---
    if drift_signals:
        lines.append("<b>\u26a0\ufe0f DRIFT ALERTS</b>")
        for ds in drift_signals[:5]:
            mkt = ds.get("market", "?")
            sig = ds.get("signal", "?")
            lines.append(f"  \u26a0\ufe0f {mkt}: {sig}")
        lines.append("")

    lines.append(SEP)
    lines.append(f"\U0001f552 Generated {datetime.now().strftime('%H:%M')}")

    return _ensure_parts("\n".join(lines))


# ---------------------------------------------------------------------------
# 2. Pre-Kickoff Update  (when delta > 2pp)
# ---------------------------------------------------------------------------


def format_pre_kickoff_update(
    deltas: list[dict[str, Any]],
    match_info: dict[str, Any] | None = None,
) -> list[str]:
    """Format probability delta notifications after lineup injection.

    Args:
        deltas: Dicts with keys: home_team, away_team, market, delta,
                probability_updated, action (UPGRADE/DOWNGRADE).
        match_info: Optional context (e.g. mins_until kickoff).

    Returns:
        List of message parts.
    """
    if not deltas:
        return []

    lines: list[str] = [
        "\U0001f504 <b>PRE-KICKOFF UPDATE</b>",
        SEP,
        "",
    ]

    for row in deltas[:10]:
        action = row.get("action", "")
        emoji = "\u2b06\ufe0f" if action == "UPGRADE" else "\u2b07\ufe0f"
        match_name = _match_name(row)
        market = row.get("market", "?")
        delta = float(row.get("delta", 0)) * 100
        new_prob = float(row.get("probability_updated", 0))

        lines.append(f"{emoji} <b>{match_name}</b>")
        lines.append(f"   {market}: {delta:+.1f}pp \u2192 {new_prob:.1%}")
        lines.append("")

    lines.append(SEP)

    return _ensure_parts("\n".join(lines))


# ---------------------------------------------------------------------------
# 3. Weekly Report  (Mon 8:00)
# ---------------------------------------------------------------------------


def format_weekly_report(
    performance: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    weekend: dict[str, Any] | None = None,
) -> list[str]:
    """Format consolidated weekly report (replaces daily monitor + drift + weekend).

    Args:
        performance: Live performance dict (from live_performance.json).
        drift: Drift report dict (from drift_report.json).
        weekend: Weekend report dict (from weekend_report JSON).

    Returns:
        List of message parts.
    """
    lines: list[str] = [
        f"\U0001f4cb <b>WEEKLY REPORT \u2022 {_today()}</b>",
        SEP,
    ]

    # --- Severity ---
    severity = _compute_severity(performance, drift)
    severity_emoji = {
        "GREEN": "\U0001f7e2",
        "YELLOW": "\U0001f7e1",
        "RED": "\U0001f534",
    }
    fallback = "\u2753"
    lines.append(f"Status: {severity_emoji.get(severity, fallback)} {severity}")
    lines.append("")

    # --- Weekend paper-trading results ---
    if weekend:
        s = weekend.get("summary", {})
        week_label = weekend.get("week", "")
        dates = weekend.get("dates", {})
        fri = dates.get("friday", "")
        sun = dates.get("sunday", "")
        date_range = f"{fri} \u2013 {sun}" if fri and sun else week_label

        pnl = s.get("pnl", 0)
        pnl_emoji = "\U0001f4b0" if pnl >= 0 else "\U0001f4c9"
        lines.append(f"{pnl_emoji} <b>Weekend ({date_range})</b>")
        lines.append(
            f"  Bets: {s.get('total_bets', 0)} | "
            f"W: {s.get('wins', 0)} L: {s.get('losses', 0)} | "
            f"Win: {s.get('win_rate', 0):.1f}%"
        )
        if s.get("staked"):
            lines.append(
                f"  Staked: {s['staked']:,.0f} PLN | "
                f"P&L: {pnl:+,.0f} PLN | ROI: {s.get('roi', 0):+.1f}%"
            )
        if s.get("unsettled", 0) > 0:
            lines.append(f"  \u26a0\ufe0f {s['unsettled']} bet(s) unsettled")

        # Per market
        per_market = weekend.get("per_market", {})
        if per_market:
            lines.append("  <b>Per Market:</b>")
            for mkt, ms in per_market.items():
                lines.append(
                    f"    {mkt}: {ms.get('wins', 0)}W/{ms.get('losses', 0)}L "
                    f"{ms.get('pnl', 0):+,.0f} PLN"
                )
        lines.append("")

    # --- Live performance (all-time) ---
    if performance:
        settled = performance.get("total_settled", 0)
        roi = performance.get("overall_roi", 0)
        pnl = performance.get("overall_pnl_units", 0)
        wr = performance.get("overall_win_rate", 0)

        lines.append("<b>\U0001f4c8 Live Performance (all-time)</b>")
        lines.append(
            f"  {settled} bets | {wr:.1f}% WR | {roi:+.1f}% ROI | {pnl:+.1f}u"
        )

        # Per-market breakdown
        markets = performance.get("markets", [])
        if markets:
            profitable = [m for m in markets if m.get("roi", 0) > 0 and m.get("total", 0) >= 10]
            struggling = [m for m in markets if m.get("roi", 0) <= 0 and m.get("total", 0) >= 10]

            if profitable:
                lines.append("  <b>Top:</b>")
                for m in sorted(profitable, key=lambda x: x.get("roi", 0), reverse=True)[:5]:
                    lines.append(
                        f"    \u2705 {m['market']}: {m['roi']:+.1f}% "
                        f"(n={m['total']})"
                    )
            if struggling:
                lines.append("  <b>Watch:</b>")
                for m in sorted(struggling, key=lambda x: x.get("roi", 0))[:3]:
                    lines.append(
                        f"    \U0001f4c9 {m['market']}: {m['roi']:+.1f}% "
                        f"(n={m['total']})"
                    )

        # Alerts
        alerts = performance.get("alerts", [])
        if alerts:
            lines.append("")
            lines.append("  <b>\U0001f6a8 Alerts:</b>")
            for alert in alerts[:5]:
                mkt = alert.get("market", "?")
                reasons = alert.get("reasons", alert.get("alert_reasons", ""))
                lines.append(f"    \U0001f6a8 {mkt}: {reasons}")
        lines.append("")

    # --- Drift status ---
    if drift:
        drift_markets = drift.get("markets", [])
        drifted = [m for m in drift_markets if m.get("status") == "drifted"]

        lines.append("<b>\U0001f50d Drift Status</b>")
        if drifted:
            lines.append(
                f"  {len(drift_markets)} checked | "
                f"\u26a0\ufe0f {len(drifted)} drifted"
            )
            for dm in drifted[:5]:
                mkt = dm.get("market", "?")
                frac = dm.get("fraction_drifted", 0)
                lines.append(f"    \u26a0\ufe0f {mkt}: {frac:.0%} features drifted")
        else:
            lines.append(
                f"  \u2705 All {len(drift_markets)} markets stable"
            )
        lines.append("")

    lines.append(SEP)

    return _ensure_parts("\n".join(lines))


# ---------------------------------------------------------------------------
# 4. Post-Optimization  (after sniper run)
# ---------------------------------------------------------------------------


def format_post_optimization(report: dict[str, Any]) -> list[str]:
    """Format post-optimization validation results.

    Real-odds markets show ROI + ECE + precision.
    Model-only markets show precision + ECE + FVA — explicitly no ROI.

    Args:
        report: Comparison report dict (from comparison_report.json).

    Returns:
        List of message parts.
    """
    if not report:
        return ["\U0001f52c <b>POST-OPTIMIZATION</b>\nNo report data."]

    run_id = report.get("source_run", "?")
    total = report.get("total_markets", 0)
    n_deploy = report.get("deploy", 0)
    n_hold = report.get("hold", 0)
    n_reject = report.get("reject", 0)
    markets = report.get("markets", [])

    lines: list[str] = [
        "\U0001f52c <b>POST-OPTIMIZATION VALIDATION</b>",
        SEP,
        f"\U0001f4ca Run #{run_id} \u2014 {total} markets analyzed",
        f"\u2705 DEPLOY: {n_deploy}  |  \u23f8 HOLD: {n_hold}  |  \u274c REJECT: {n_reject}",
        "",
    ]

    deploy_markets = [m for m in markets if m.get("verdict") == "DEPLOY"]
    hold_markets = [m for m in markets if m.get("verdict") == "HOLD"]
    reject_markets = [m for m in markets if m.get("verdict") == "REJECT"]

    # --- Deploy ---
    if deploy_markets:
        lines.append("<b>\u2705 DEPLOY</b>")
        for m in deploy_markets:
            lines.append(_format_optimization_market(m))
        lines.append("")

    # --- Hold ---
    if hold_markets:
        lines.append("<b>\u23f8 HOLD</b>")
        for m in hold_markets[:5]:
            lines.append(_format_optimization_market(m))
        if len(hold_markets) > 5:
            lines.append(f"  ...+{len(hold_markets) - 5} more")
        lines.append("")

    # --- Reject ---
    if reject_markets:
        lines.append("<b>\u274c REJECT</b>")
        for m in reject_markets[:5]:
            mkt = m.get("market", "?")
            violations = m.get("violations", "")
            if violations:
                lines.append(f"  {mkt}: {violations}")
            else:
                lines.append(_format_optimization_market(m))
        if len(reject_markets) > 5:
            lines.append(f"  ...+{len(reject_markets) - 5} more")
        lines.append("")

    lines.append(SEP)

    return _ensure_parts("\n".join(lines))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_optimization_market(m: dict[str, Any]) -> str:
    """Format a single market line for post-optimization, market-aware."""
    mkt = m.get("market", "?")
    ece = m.get("ece", 0)
    precision = m.get("precision", 0)
    n_bets = m.get("n_bets", 0)
    n_features = m.get("n_features", m.get("features", ""))
    adv_auc = m.get("adversarial_auc", m.get("adv_auc", 0))

    if is_real_odds_market(mkt):
        roi = m.get("roi", 0)
        roi_delta = m.get("roi_delta")
        delta_str = f" ({roi_delta:+.1f}pp)" if roi_delta is not None else ""
        return (
            f"  {mkt}: ROI {roi:+.1f}%{delta_str}, "
            f"ECE {ece:.3f}, prec {precision:.0f}%, "
            f"{n_features}f, adv {adv_auc:.2f}"
        )
    else:
        return (
            f"  {mkt}: prec {precision:.0f}%, ECE {ece:.3f}, "
            f"{n_features}f, adv {adv_auc:.2f} "
            f"<i>(no ROI \u2014 estimated odds)</i>"
        )


def _match_name(rec: dict[str, Any], max_len: int = 35) -> str:
    """Build 'Home vs Away' string, truncated."""
    home = rec.get("home_team", "?")
    away = rec.get("away_team", "?")
    name = f"{home} vs {away}"
    return name[:max_len] if len(name) > max_len else name


def _pct(value: Any) -> str:
    """Convert probability (0-1 float or 0-100) to display percentage."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "?"
    if v <= 1.0:
        v *= 100
    return f"{v:.1f}"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _compute_severity(
    performance: dict[str, Any] | None,
    drift: dict[str, Any] | None,
) -> str:
    """Derive GREEN/YELLOW/RED severity from performance + drift."""
    has_alerts = bool(performance and performance.get("alerts"))
    has_drift = bool(
        drift
        and any(m.get("status") == "drifted" for m in drift.get("markets", []))
    )
    roi = (performance or {}).get("overall_roi", 0)

    if has_alerts or roi < -10:
        return "RED"
    if has_drift or roi < 0:
        return "YELLOW"
    return "GREEN"


def _ensure_parts(text: str, max_len: int = 4096) -> list[str]:
    """Split text into parts at newline boundaries if needed."""
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in text.split("\n"):
        line_len = len(line) + 1
        if current_len + line_len > max_len and current_lines:
            parts.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        parts.append("\n".join(current_lines))

    return parts
