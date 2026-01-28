"""
Sample Weighting Utilities for Model Training

Implements time-decayed observation weighting inspired by the retail demand forecasting
paper "One Global Model, Many Behaviors" (VN2 Winner).

Key concepts:
- Recent observations weighted higher than older ones
- Exponential decay: weight = exp(-decay_rate * days_ago)
- Configurable decay rate (half-life parameter)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from datetime import datetime


def calculate_time_decay_weights(
    dates: Union[np.ndarray, pd.Series],
    decay_rate: float = 0.002,
    reference_date: Optional[datetime] = None,
    min_weight: float = 0.1,
) -> np.ndarray:
    """
    Calculate time-decayed sample weights based on observation dates.

    Recent observations get weight close to 1.0, older observations get lower weights
    following exponential decay. This helps the model focus more on recent patterns
    while still learning from historical data.

    Args:
        dates: Array of dates for each observation (can be datetime or string)
        decay_rate: Decay rate per day. Higher = faster decay.
                   Common values:
                   - 0.001: ~2 year half-life (gentle decay)
                   - 0.002: ~1 year half-life (moderate decay)
                   - 0.005: ~4.5 month half-life (aggressive decay)
        reference_date: Date to measure days from (default: most recent date in data)
        min_weight: Minimum weight for oldest observations (prevents zero weights)

    Returns:
        Array of weights in [min_weight, 1.0]

    Example:
        >>> dates = pd.to_datetime(['2024-01-01', '2023-06-01', '2023-01-01'])
        >>> weights = calculate_time_decay_weights(dates, decay_rate=0.002)
        >>> # More recent dates get higher weights
    """
    # Convert to DatetimeIndex
    dates = pd.to_datetime(dates)
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    # Use most recent date as reference if not provided
    if reference_date is None:
        reference_date = dates.max()
    else:
        reference_date = pd.to_datetime(reference_date)

    # Calculate days ago from reference date
    # TimedeltaIndex doesn't have .dt accessor, use .days directly
    timedelta_index = reference_date - dates
    days_ago = timedelta_index.days.values
    days_ago = np.maximum(days_ago, 0)  # Ensure non-negative

    # Exponential decay: weight = exp(-decay_rate * days_ago)
    weights = np.exp(-decay_rate * days_ago)

    # Apply minimum weight floor
    weights = np.maximum(weights, min_weight)

    return weights


def calculate_tiered_weights(
    dates: Union[np.ndarray, pd.Series],
    tier_days: tuple = (365, 730),
    tier_weights: tuple = (1.0, 0.5, 0.25),
    reference_date: Optional[datetime] = None,
) -> np.ndarray:
    """
    Calculate tiered sample weights based on age categories.

    Alternative to exponential decay - uses discrete weight tiers.
    Inspired by the original paper which used 1.0/0.5/0.25 for year-based tiers.

    Args:
        dates: Array of observation dates
        tier_days: Day thresholds for tier boundaries (default: 1 year, 2 years)
        tier_weights: Weights for each tier (recent, middle, oldest)
        reference_date: Date to measure from (default: most recent)

    Returns:
        Array of tier-based weights

    Example:
        >>> dates = pd.to_datetime(['2024-01-01', '2023-06-01', '2022-01-01'])
        >>> # With default tiers: [1.0, 0.5, 0.25]
        >>> weights = calculate_tiered_weights(dates)
    """
    # Convert to DatetimeIndex
    dates = pd.to_datetime(dates)
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    if reference_date is None:
        reference_date = dates.max()
    else:
        reference_date = pd.to_datetime(reference_date)

    # TimedeltaIndex doesn't have .dt accessor, use .days directly
    timedelta_index = reference_date - dates
    days_ago = timedelta_index.days.values
    days_ago = np.maximum(days_ago, 0)

    weights = np.full(len(days_ago), tier_weights[-1])  # Default to oldest tier weight

    # Process tiers from newest to oldest, only updating unassigned samples
    # tier_days are thresholds (e.g., 365, 730)
    # tier_weights[0] is for days < tier_days[0]
    # tier_weights[1] is for tier_days[0] <= days < tier_days[1]
    # tier_weights[-1] is for days >= tier_days[-1]

    # Start with the second-to-last tier and work backwards
    prev_threshold = float('inf')
    for threshold, weight in reversed(list(zip(tier_days, tier_weights[:-1]))):
        # Apply weight only to samples in this tier's range
        mask = (days_ago < threshold)
        weights[mask] = weight
        prev_threshold = threshold

    return weights


def decay_rate_from_half_life(half_life_days: float) -> float:
    """
    Calculate decay rate from desired half-life in days.

    Half-life is the number of days after which weight drops to 0.5.

    Args:
        half_life_days: Number of days for weight to decay to 0.5

    Returns:
        Decay rate parameter for exponential decay

    Example:
        >>> rate = decay_rate_from_half_life(365)  # 1-year half-life
        >>> rate  # ~0.0019
    """
    return np.log(2) / half_life_days


def half_life_from_decay_rate(decay_rate: float) -> float:
    """
    Calculate half-life in days from decay rate.

    Args:
        decay_rate: Exponential decay rate

    Returns:
        Half-life in days
    """
    return np.log(2) / decay_rate


def get_recommended_decay_rate(sport: str = "football") -> float:
    """
    Get recommended decay rate based on sport and typical seasonality.

    Football/soccer has clear seasonal patterns with ~300 day seasons.
    Recommended decay gives meaningful weight to last 2-3 seasons.

    Args:
        sport: Sport type (currently only 'football' supported)

    Returns:
        Recommended decay rate
    """
    # Football: ~300 match days per season
    # We want ~2 seasons to have meaningful weight (50% at 1 season)
    if sport == "football":
        return 0.002  # ~346 day half-life (roughly 1 season)

    return 0.002  # Default


def normalize_weights(weights: np.ndarray, target_mean: float = 1.0) -> np.ndarray:
    """
    Normalize weights to have a specific mean.

    Useful when combining sample weights with class weights or when
    frameworks expect weights to average to 1.0.

    Args:
        weights: Raw sample weights
        target_mean: Desired mean of normalized weights

    Returns:
        Normalized weights with specified mean
    """
    current_mean = weights.mean()
    if current_mean > 0:
        return weights * (target_mean / current_mean)
    return weights
