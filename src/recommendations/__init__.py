"""
Recommendations module for generating and tracking betting recommendations.
"""
from src.recommendations.generator import (
    RECOMMENDATION_COLUMNS,
    MARKETS,
    MARKET_SIDES,
    Recommendation,
    RecommendationGenerator,
    load_recommendations,
    update_result,
    calculate_performance,
)

__all__ = [
    'RECOMMENDATION_COLUMNS',
    'MARKETS',
    'MARKET_SIDES',
    'Recommendation',
    'RecommendationGenerator',
    'load_recommendations',
    'update_result',
    'calculate_performance',
]
