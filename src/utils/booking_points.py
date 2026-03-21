"""Booking points computation for card markets.

Bookmaker convention:
- Yellow card = 1 booking point
- Red card = 2 booking points
- 2nd yellow -> red: 1st yellow (1pt) + red (2pt) = 3 total.
  The 2nd yellow is NOT counted separately (absorbed by the red).
"""

from typing import Union

import numpy as np
import pandas as pd


def compute_booking_points_from_stats(
    yellows: Union[int, float, pd.Series, np.ndarray],
    reds: Union[int, float, pd.Series, np.ndarray],
) -> Union[int, float, pd.Series, np.ndarray]:
    """Compute booking points from aggregate stats (no player-level data).

    Uses yellows + reds * 2 as best available approximation.
    Slightly overcounts (~0.07/match) for 2nd-yellow-to-red incidents
    since the absorbed 2nd yellow can't be detected without player_id.
    Still far more accurate than raw yellows + reds (undercounts straight reds).

    Args:
        yellows: Yellow card count(s). Scalar or array-like.
        reds: Red card count(s). Scalar or array-like.

    Returns:
        Booking points in the same type as input.
    """
    return yellows + reds * 2


def compute_booking_points_from_events(
    card_events: pd.DataFrame,
    home_team_col: str = "is_home",
) -> pd.DataFrame:
    """Compute exact booking points from card events with player-level data.

    Detects 2nd-yellow-to-red incidents per player and subtracts the
    absorbed 2nd yellow that bookmakers don't count.

    Args:
        card_events: DataFrame with columns:
            - fixture_id: match identifier
            - player_id: player identifier (for 2Y->R detection)
            - detail: 'Yellow Card' or 'Red Card'
            - is_home (or home_team_col): bool, True if home team card
        home_team_col: name of the boolean column indicating home team

    Returns:
        DataFrame with columns [fixture_id, total_cards, home_cards, away_cards]
        where values represent booking points.
    """
    if card_events.empty:
        return pd.DataFrame(columns=["fixture_id", "total_cards", "home_cards", "away_cards"])

    df = card_events.copy()
    df["is_yellow"] = (df["detail"] == "Yellow Card").astype(int)
    df["is_red"] = (df["detail"] == "Red Card").astype(int)

    # Per-player card counts within each fixture + side
    player_cards = (
        df.groupby(["fixture_id", "player_id", home_team_col])
        .agg(n_yellows=("is_yellow", "sum"), n_reds=("is_red", "sum"))
        .reset_index()
    )

    # Detect 2Y->R: player has at least 1 yellow AND at least 1 red in same fixture
    player_cards["is_2y_r"] = (
        (player_cards["n_yellows"] >= 1) & (player_cards["n_reds"] >= 1)
    ).astype(int)

    # Per-player booking points: yellows + reds*2 - absorbed_2nd_yellow
    player_cards["booking_pts"] = (
        player_cards["n_yellows"]
        + player_cards["n_reds"] * 2
        - player_cards["is_2y_r"]  # subtract 1 absorbed 2nd yellow per 2Y->R
    )

    # Aggregate per fixture + side
    side_totals = (
        player_cards.groupby(["fixture_id", home_team_col])["booking_pts"]
        .sum()
        .reset_index()
    )

    # Pivot to home/away
    home = side_totals[side_totals[home_team_col] == True][  # noqa: E712
        ["fixture_id", "booking_pts"]
    ].rename(columns={"booking_pts": "home_cards"})

    away = side_totals[side_totals[home_team_col] == False][  # noqa: E712
        ["fixture_id", "booking_pts"]
    ].rename(columns={"booking_pts": "away_cards"})

    result = home.merge(away, on="fixture_id", how="outer")
    result["home_cards"] = result["home_cards"].fillna(0).astype(int)
    result["away_cards"] = result["away_cards"].fillna(0).astype(int)
    result["total_cards"] = result["home_cards"] + result["away_cards"]

    return result
