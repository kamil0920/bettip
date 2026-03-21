"""Tests for src.notifications — market classification, formatters, and Telegram client."""

from __future__ import annotations

import pytest

from src.notifications.market_utils import (
    REAL_ODDS_MARKETS,
    classify_recommendations,
    is_real_odds_market,
)
from src.notifications.formatters import (
    format_match_day_digest,
    format_post_optimization,
    format_pre_kickoff_update,
    format_weekly_report,
)
from src.notifications.telegram import TelegramNotifier, _split_message


# ---------------------------------------------------------------------------
# market_utils
# ---------------------------------------------------------------------------


class TestIsRealOddsMarket:
    def test_real_odds_markets(self):
        for mkt in ("home_win", "away_win", "over25", "under25", "btts",
                     "home_win_h1", "away_win_h1"):
            assert is_real_odds_market(mkt) is True, f"{mkt} should be real-odds"

    def test_estimated_odds_markets(self):
        for mkt in ("corners_over_95", "shots", "fouls", "cards",
                     "corners_under_105", "fouls_over_265"):
            assert is_real_odds_market(mkt) is False, f"{mkt} should be model-only"

    def test_unknown_market(self):
        assert is_real_odds_market("unknown_market") is False

    def test_real_odds_markets_is_frozenset(self):
        assert isinstance(REAL_ODDS_MARKETS, frozenset)


class TestClassifyRecommendations:
    def test_split_by_market_name(self):
        recs = [
            {"market": "home_win", "edge": 10, "probability": 0.65},
            {"market": "corners_over_95", "edge": 20, "probability": 0.70},
            {"market": "away_win", "edge": 5, "probability": 0.55},
        ]
        real, model = classify_recommendations(recs)
        assert len(real) == 2
        assert len(model) == 1
        assert all(r["market"] in REAL_ODDS_MARKETS for r in real)
        assert model[0]["market"] == "corners_over_95"

    def test_odds_verified_overrides_market_name(self):
        recs = [
            {"market": "corners_over_95", "odds_verified": "true", "edge": 5, "probability": 0.6},
            {"market": "home_win", "odds_verified": "false", "edge": 10, "probability": 0.7},
        ]
        real, model = classify_recommendations(recs)
        assert len(real) == 1
        assert real[0]["market"] == "corners_over_95"
        assert len(model) == 1
        assert model[0]["market"] == "home_win"

    def test_real_odds_sorted_by_edge(self):
        recs = [
            {"market": "home_win", "edge": 5, "probability": 0.6},
            {"market": "away_win", "edge": 15, "probability": 0.5},
        ]
        real, _ = classify_recommendations(recs)
        assert real[0]["edge"] == 15  # Higher edge first

    def test_model_only_sorted_by_probability(self):
        recs = [
            {"market": "corners_over_95", "probability": 0.6},
            {"market": "fouls_over_235", "probability": 0.8},
        ]
        _, model = classify_recommendations(recs)
        assert model[0]["probability"] == 0.8  # Higher prob first

    def test_empty_input(self):
        real, model = classify_recommendations([])
        assert real == []
        assert model == []

    def test_fallback_to_bet_type(self):
        recs = [{"bet_type": "home_win", "edge": 5, "probability": 0.6}]
        real, model = classify_recommendations(recs)
        assert len(real) == 1


# ---------------------------------------------------------------------------
# formatters — match_day_digest
# ---------------------------------------------------------------------------


class TestFormatMatchDayDigest:
    def test_no_recs(self):
        parts = format_match_day_digest([])
        assert len(parts) == 1
        assert "No picks" in parts[0]

    def test_real_odds_shows_edge(self):
        recs = [
            {"market": "home_win", "home_team": "Arsenal", "away_team": "Chelsea",
             "edge": 12, "probability": 0.65, "odds": 1.85},
        ]
        parts = format_match_day_digest(recs, total_matches=5)
        text = "\n".join(parts)
        assert "REAL ODDS" in text
        assert "Edge +12%" in text
        assert "Prob 65.0%" in text

    def test_model_only_no_edge(self):
        recs = [
            {"market": "corners_over_95", "home_team": "Arsenal", "away_team": "Chelsea",
             "edge": 20, "probability": 0.70, "odds": 1.90},
        ]
        parts = format_match_day_digest(recs, total_matches=3)
        text = "\n".join(parts)
        assert "MODEL ONLY" in text
        assert "Prob 70.0%" in text
        # Edge should NOT appear in model-only section
        assert "Edge" not in text.split("MODEL ONLY")[1]

    def test_drift_alerts_included(self):
        recs = [{"market": "home_win", "home_team": "A", "away_team": "B",
                 "edge": 5, "probability": 0.6}]
        drift = [{"market": "home_win", "signal": "ECE drift +0.03"}]
        parts = format_match_day_digest(recs, drift_signals=drift)
        text = "\n".join(parts)
        assert "DRIFT ALERTS" in text
        assert "ECE drift" in text

    def test_mixed_markets(self):
        recs = [
            {"market": "home_win", "home_team": "A", "away_team": "B",
             "edge": 15, "probability": 0.7},
            {"market": "corners_over_95", "home_team": "C", "away_team": "D",
             "probability": 0.75},
        ]
        parts = format_match_day_digest(recs, total_matches=10)
        text = "\n".join(parts)
        assert "REAL ODDS" in text
        assert "MODEL ONLY" in text


# ---------------------------------------------------------------------------
# formatters — pre_kickoff_update
# ---------------------------------------------------------------------------


class TestFormatPreKickoffUpdate:
    def test_empty_deltas(self):
        parts = format_pre_kickoff_update([])
        assert parts == []

    def test_upgrade(self):
        deltas = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "market": "home_win",
             "delta": 0.05, "probability_updated": 0.70, "action": "UPGRADE"},
        ]
        parts = format_pre_kickoff_update(deltas)
        text = "\n".join(parts)
        assert "PRE-KICKOFF" in text
        assert "+5.0pp" in text
        assert "70.0%" in text


# ---------------------------------------------------------------------------
# formatters — weekly_report
# ---------------------------------------------------------------------------


class TestFormatWeeklyReport:
    def test_all_none(self):
        parts = format_weekly_report()
        text = "\n".join(parts)
        assert "WEEKLY REPORT" in text
        assert "GREEN" in text  # No alerts, no drift → GREEN

    def test_with_performance(self):
        perf = {
            "total_settled": 100,
            "overall_roi": 12.5,
            "overall_pnl_units": 42.0,
            "overall_win_rate": 66.0,
            "markets": [
                {"market": "home_win", "roi": 15.0, "total": 30},
                {"market": "fouls", "roi": -5.0, "total": 20},
            ],
            "alerts": [],
        }
        parts = format_weekly_report(performance=perf)
        text = "\n".join(parts)
        assert "100 bets" in text
        assert "+12.5% ROI" in text

    def test_red_severity_on_alerts(self):
        perf = {
            "total_settled": 50,
            "overall_roi": -15.0,
            "overall_pnl_units": -10.0,
            "overall_win_rate": 40.0,
            "alerts": [{"market": "home_win", "reasons": "ECE drift"}],
        }
        parts = format_weekly_report(performance=perf)
        text = "\n".join(parts)
        assert "RED" in text

    def test_with_drift(self):
        drift = {
            "markets": [
                {"market": "home_win", "status": "ok"},
                {"market": "btts", "status": "drifted", "fraction_drifted": 0.3},
            ]
        }
        parts = format_weekly_report(drift=drift)
        text = "\n".join(parts)
        assert "drifted" in text.lower()
        assert "YELLOW" in text

    def test_with_weekend(self):
        weekend = {
            "week": "2026-W12",
            "dates": {"friday": "2026-03-20", "saturday": "2026-03-21", "sunday": "2026-03-22"},
            "summary": {
                "total_bets": 8, "wins": 5, "losses": 3, "pushes": 0,
                "win_rate": 62.5, "staked": 800, "pnl": 150, "roi": 18.8,
                "unsettled": 0,
            },
            "per_market": {"home_win": {"wins": 3, "losses": 1, "pnl": 100}},
        }
        parts = format_weekly_report(weekend=weekend)
        text = "\n".join(parts)
        assert "Weekend" in text
        assert "8" in text  # total bets
        assert "+150 PLN" in text


# ---------------------------------------------------------------------------
# formatters — post_optimization
# ---------------------------------------------------------------------------


class TestFormatPostOptimization:
    def test_empty_report(self):
        parts = format_post_optimization({})
        assert len(parts) == 1
        assert "No report data" in parts[0]

    def test_real_odds_shows_roi(self):
        report = {
            "source_run": "12345",
            "total_markets": 2,
            "deploy": 1,
            "hold": 1,
            "reject": 0,
            "markets": [
                {"market": "home_win", "verdict": "DEPLOY", "roi": 25.0,
                 "ece": 0.035, "precision": 85, "n_features": 8, "adversarial_auc": 0.78},
                {"market": "corners_over_95", "verdict": "HOLD", "roi": 50.0,
                 "ece": 0.04, "precision": 80, "n_features": 6, "adversarial_auc": 0.72},
            ],
        }
        parts = format_post_optimization(report)
        text = "\n".join(parts)

        # Real-odds market should show ROI
        assert "ROI +25.0%" in text

        # Model-only market should show "no ROI" note
        assert "no ROI" in text
        # And should NOT show a numeric ROI value for corners
        assert "corners_over_95: ROI" not in text

    def test_market_aware_formatting(self):
        report = {
            "source_run": "999",
            "total_markets": 1,
            "deploy": 1,
            "hold": 0,
            "reject": 0,
            "markets": [
                {"market": "fouls_over_235", "verdict": "DEPLOY",
                 "ece": 0.02, "precision": 90, "n_features": 5, "adversarial_auc": 0.70},
            ],
        }
        parts = format_post_optimization(report)
        text = "\n".join(parts)
        assert "estimated odds" in text
        assert "prec 90%" in text


# ---------------------------------------------------------------------------
# telegram — message splitting
# ---------------------------------------------------------------------------


class TestMessageSplitting:
    def test_short_message_no_split(self):
        parts = _split_message("hello world", 4096)
        assert parts == ["hello world"]

    def test_long_message_splits_at_newline(self):
        # 10 lines of 500 chars each = 5000+ chars
        lines = [f"Line {i}: {'x' * 490}" for i in range(10)]
        text = "\n".join(lines)
        assert len(text) > 4096

        parts = _split_message(text, 4096)
        assert len(parts) >= 2
        for part in parts:
            assert len(part) <= 4096

        # Reassembled content should match original
        reassembled = "\n".join(parts)
        assert reassembled == text

    def test_empty_message(self):
        parts = _split_message("", 4096)
        assert parts == [""]


# ---------------------------------------------------------------------------
# telegram — TelegramNotifier
# ---------------------------------------------------------------------------


class TestTelegramNotifier:
    def test_unconfigured_dry_run(self):
        notifier = TelegramNotifier(token="", chat_id="")
        assert notifier.is_configured is False
        # send_message should return False (dry run)
        result = notifier.send_message("test")
        assert result is False

    def test_configured(self):
        notifier = TelegramNotifier(token="fake:token", chat_id="12345")
        assert notifier.is_configured is True

    def test_send_parts_empty(self):
        notifier = TelegramNotifier(token="", chat_id="")
        assert notifier.send_parts([]) is True

    def test_env_fallback(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        notifier = TelegramNotifier()
        assert notifier.is_configured is False
