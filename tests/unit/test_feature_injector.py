"""Tests for external feature injection at inference time."""
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from pathlib import Path


class TestRefereeFeatureInjection:
    """Test referee feature injection."""

    @pytest.fixture
    def injector(self, tmp_path):
        """Create injector with mock referee cache."""
        from src.ml.feature_injector import ExternalFeatureInjector

        # Create mock referee stats cache
        referee_cache = pd.DataFrame({
            'referee_name': ['Michael Oliver', 'Anthony Taylor', 'New Referee'],
            'matches': [50, 30, 3],  # 3 matches is below MIN_REFEREE_MATCHES
            'total_yellows': [200, 120, 12],
            'total_reds': [10, 3, 0],
            'total_fouls': [1100, 660, 66],
            'total_corners': [520, 310, 30],
            'home_wins': [23, 15, 1],
            'draws': [12, 7, 1],
            'away_wins': [15, 8, 1],
            'total_goals': [140, 85, 8],
        })
        cache_path = tmp_path / 'referee_stats.parquet'
        referee_cache.to_parquet(cache_path)
        return ExternalFeatureInjector(
            referee_cache_path=str(cache_path),
            enable_weather=False,  # Disable weather for referee-only tests
        )

    def test_inject_known_referee_features(self, injector):
        """Known referee should inject actual stats."""
        features_df = pd.DataFrame({'fixture_id': [1]})

        result = injector._inject_referee_features(features_df.copy(), 'Michael Oliver')

        # Michael Oliver: 50 matches, 210 cards (200 yellows + 10 reds)
        # ref_cards_avg = 210/50 = 4.2
        assert result['ref_cards_avg'].iloc[0] == pytest.approx(4.2, rel=0.1)
        # ref_fouls_avg = 1100/50 = 22.0
        assert result['ref_fouls_avg'].iloc[0] == pytest.approx(22.0, rel=0.1)
        # ref_corners_avg = 520/50 = 10.4
        assert result['ref_corners_avg'].iloc[0] == pytest.approx(10.4, rel=0.1)
        # ref_matches = 50
        assert result['ref_matches'].iloc[0] == 50

    def test_inject_unknown_referee_uses_defaults(self, injector):
        """Unknown referee should use league average defaults."""
        features_df = pd.DataFrame({'fixture_id': [1]})

        result = injector._inject_referee_features(features_df.copy(), 'Unknown Referee')

        # Should use REFEREE_DEFAULTS
        assert result['ref_cards_avg'].iloc[0] == pytest.approx(4.2, rel=0.1)
        assert result['ref_fouls_avg'].iloc[0] == pytest.approx(22.0, rel=0.1)
        assert result['ref_matches'].iloc[0] == 0

    def test_inject_none_referee_uses_defaults(self, injector):
        """No referee assigned should use defaults."""
        features_df = pd.DataFrame({'fixture_id': [1]})

        result = injector._inject_referee_features(features_df.copy(), None)

        assert result['ref_cards_avg'].iloc[0] == pytest.approx(4.2, rel=0.1)
        assert result['ref_matches'].iloc[0] == 0

    def test_inject_referee_with_insufficient_matches_uses_defaults(self, injector):
        """Referee with < min_matches should use defaults."""
        features_df = pd.DataFrame({'fixture_id': [1]})

        # 'New Referee' has only 3 matches (below MIN_REFEREE_MATCHES=5)
        result = injector._inject_referee_features(features_df.copy(), 'New Referee')

        # Should use defaults
        assert result['ref_cards_avg'].iloc[0] == pytest.approx(4.2, rel=0.1)
        assert result['ref_matches'].iloc[0] == 0

    def test_referee_bias_calculation(self, injector):
        """Referee bias features should be calculated correctly."""
        features_df = pd.DataFrame({'fixture_id': [1]})

        result = injector._inject_referee_features(features_df.copy(), 'Michael Oliver')

        # home_win_pct = 23/50 = 0.46, default = 0.46, bias = 0
        assert result['ref_home_bias'].iloc[0] == pytest.approx(0.0, abs=0.01)
        # cards_avg = 4.2, default = 4.2, bias = 0
        assert result['ref_cards_bias'].iloc[0] == pytest.approx(0.0, abs=0.1)


class TestWeatherFeatureInjection:
    """Test weather feature injection."""

    @pytest.fixture
    def injector(self, tmp_path):
        """Create injector with mocked weather collector."""
        from src.ml.feature_injector import ExternalFeatureInjector

        return ExternalFeatureInjector(
            referee_cache_path=str(tmp_path / 'empty.parquet'),
            enable_referee=False,  # Disable referee for weather-only tests
        )

    def test_inject_weather_from_forecast(self, injector):
        """Weather forecast should inject features."""
        # Mock the weather collector
        mock_collector = Mock()
        mock_collector.fetch_forecast.return_value = {
            'temperature': 12.5,
            'precipitation': 2.1,
            'wind_speed': 25.0,
            'humidity': 90,  # > 85 threshold
            'weather_code': 61,  # Light rain
        }
        injector.weather_collector = mock_collector

        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {
            'venue_city': 'London',
            'kickoff': datetime(2026, 2, 5, 15, 0),
        }

        result = injector._inject_weather_features(features_df.copy(), match_info)

        assert result['weather_temp'].iloc[0] == 12.5
        assert result['weather_precip'].iloc[0] == 2.1
        assert result['weather_is_rainy'].iloc[0] == 1  # precip > 0.5mm
        assert result['weather_is_windy'].iloc[0] == 1  # wind > 20 km/h
        assert result['weather_high_humidity'].iloc[0] == 1  # humidity > 85

    def test_inject_weather_unavailable_uses_defaults(self, injector):
        """Missing weather should use neutral defaults."""
        mock_collector = Mock()
        mock_collector.fetch_forecast.return_value = None
        injector.weather_collector = mock_collector

        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {'venue_city': 'Unknown City', 'kickoff': datetime.now()}

        result = injector._inject_weather_features(features_df.copy(), match_info)

        assert result['weather_temp'].iloc[0] == 15.0  # Neutral default
        assert result['weather_precip'].iloc[0] == 0.0
        assert result['weather_is_rainy'].iloc[0] == 0
        assert result['weather_adverse_score'].iloc[0] == 0

    def test_inject_weather_no_city_uses_defaults(self, injector):
        """Missing city should use defaults."""
        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {'kickoff': datetime.now()}  # No venue_city

        result = injector._inject_weather_features(features_df.copy(), match_info)

        assert result['weather_temp'].iloc[0] == 15.0

    def test_weather_adverse_score_calculation(self, injector):
        """Adverse weather score should sum conditions."""
        mock_collector = Mock()
        mock_collector.fetch_forecast.return_value = {
            'temperature': 2.0,  # extreme_cold
            'precipitation': 6.0,  # is_rainy + heavy_rain
            'wind_speed': 40.0,  # is_windy + very_windy
            'humidity': 90,  # high_humidity
            'weather_code': 0,
        }
        injector.weather_collector = mock_collector

        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {
            'venue_city': 'Manchester',
            'kickoff': datetime(2026, 2, 5, 15, 0),
        }

        result = injector._inject_weather_features(features_df.copy(), match_info)

        # Score = rainy(1) + windy(1) + high_humidity(1) + extreme_cold(1) = 4
        assert result['weather_adverse_score'].iloc[0] == 4


class TestFullInjectionPipeline:
    """Test complete injection workflow."""

    @pytest.fixture
    def injector_with_data(self, tmp_path):
        """Create injector with real-like test data."""
        from src.ml.feature_injector import ExternalFeatureInjector

        referee_cache = pd.DataFrame({
            'referee_name': ['Michael Oliver'],
            'matches': [50],
            'total_yellows': [200],
            'total_reds': [10],
            'total_fouls': [1100],
            'total_corners': [520],
            'home_wins': [23],
            'draws': [12],
            'away_wins': [15],
            'total_goals': [140],
        })
        cache_path = tmp_path / 'referee_stats.parquet'
        referee_cache.to_parquet(cache_path)
        return ExternalFeatureInjector(referee_cache_path=str(cache_path))

    def test_full_injection_adds_all_features(self, injector_with_data):
        """Full injection should add referee AND weather features."""
        mock_collector = Mock()
        mock_collector.fetch_forecast.return_value = {
            'temperature': 8.0,
            'precipitation': 5.0,
            'wind_speed': 30.0,
            'humidity': 90,
            'weather_code': 63,
        }
        injector_with_data.weather_collector = mock_collector

        # Input: features without referee/weather
        features_df = pd.DataFrame({
            'fixture_id': [12345],
            'home_form_goals_scored': [1.8],
            'away_form_goals_scored': [1.2],
        })

        match_info = {
            'referee': 'Michael Oliver',
            'venue_city': 'Manchester',
            'kickoff': datetime(2026, 2, 8, 15, 0),
        }

        result = injector_with_data.inject_features(features_df, match_info)

        # Original features preserved
        assert 'home_form_goals_scored' in result.columns
        assert 'away_form_goals_scored' in result.columns

        # Referee features added
        assert 'ref_cards_avg' in result.columns
        assert 'ref_fouls_avg' in result.columns
        assert result['ref_cards_avg'].iloc[0] == pytest.approx(4.2, rel=0.1)

        # Weather features added
        assert 'weather_temp' in result.columns
        assert 'weather_is_rainy' in result.columns
        assert result['weather_is_rainy'].iloc[0] == 1  # 5mm precip

    def test_injection_handles_empty_match_info(self, injector_with_data):
        """Empty match_info should still work with defaults."""
        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {}

        result = injector_with_data.inject_features(features_df, match_info)

        # Should have defaults
        assert 'ref_cards_avg' in result.columns
        assert 'weather_temp' in result.columns
        assert result['ref_matches'].iloc[0] == 0  # Default

    def test_injection_preserves_existing_features(self, injector_with_data):
        """Injection should not modify unrelated existing features."""
        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_elo': [1500],
            'away_elo': [1480],
            'home_form_points': [2.3],
        })

        result = injector_with_data.inject_features(features_df, {})

        # Original features should be unchanged
        assert result['home_elo'].iloc[0] == 1500
        assert result['away_elo'].iloc[0] == 1480
        assert result['home_form_points'].iloc[0] == 2.3


class TestWeatherCollectorForecast:
    """Test weather collector forecast extension."""

    def test_fetch_forecast_known_city(self):
        """Known city should return forecast data."""
        from src.data_collection.weather_collector import WeatherCollector

        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'hourly': {
                    'time': ['2026-02-05T15:00'],
                    'temperature_2m': [12.5],
                    'precipitation': [0.5],
                    'wind_speed_10m': [15.0],
                    'relative_humidity_2m': [70],
                    'weather_code': [3],
                }
            }
            mock_get.return_value = mock_response

            collector = WeatherCollector()
            result = collector.fetch_forecast('London', datetime(2026, 2, 5, 15, 0))

            assert result is not None
            assert result['temperature'] == 12.5
            assert result['wind_speed'] == 15.0

    def test_fetch_forecast_unknown_city_returns_none(self):
        """Unknown city should return None."""
        from src.data_collection.weather_collector import WeatherCollector

        collector = WeatherCollector()
        result = collector.fetch_forecast('Unknown City XYZ', datetime.now())

        assert result is None


class TestWeatherFlags:
    """Test WMO weather code to flag conversion."""

    @pytest.fixture
    def injector(self, tmp_path):
        """Create injector for testing."""
        from src.ml.feature_injector import ExternalFeatureInjector
        return ExternalFeatureInjector(
            referee_cache_path=str(tmp_path / 'empty.parquet'),
            enable_weather=False,
        )

    def test_clear_weather_code(self, injector):
        """Clear sky codes should set is_clear flag."""
        flags = injector._get_weather_flags(0)
        assert flags['weather_is_clear'] == 1
        assert flags['weather_is_foggy'] == 0
        assert flags['weather_is_stormy'] == 0

    def test_fog_weather_code(self, injector):
        """Fog codes should set is_foggy flag."""
        flags = injector._get_weather_flags(45)
        assert flags['weather_is_clear'] == 0
        assert flags['weather_is_foggy'] == 1
        assert flags['weather_is_stormy'] == 0

    def test_storm_weather_code(self, injector):
        """Storm codes should set is_stormy flag."""
        flags = injector._get_weather_flags(95)
        assert flags['weather_is_clear'] == 0
        assert flags['weather_is_foggy'] == 0
        assert flags['weather_is_stormy'] == 1

    def test_nan_weather_code_uses_clear_default(self, injector):
        """NaN weather code should default to clear."""
        import numpy as np
        flags = injector._get_weather_flags(np.nan)
        assert flags['weather_is_clear'] == 1


class TestLineupFeatureInjection:
    """Test lineup feature injection."""

    @pytest.fixture
    def injector(self, tmp_path):
        """Create injector with mock player stats cache."""
        from src.ml.feature_injector import ExternalFeatureInjector

        # Create mock player stats cache with known players
        player_cache = pd.DataFrame({
            'player_id': [101, 102, 103, 104, 105, 201, 202, 203, 204, 205],
            'player_name': [
                'Salah', 'Nunez', 'Diaz', 'Szoboszlai', 'Mac Allister',  # Home
                'Haaland', 'De Bruyne', 'Foden', 'Rodri', 'Stones',  # Away
            ],
            'avg_rating': [7.8, 7.2, 7.0, 7.1, 7.3, 8.2, 7.9, 7.4, 7.6, 7.0],
            'total_minutes': [2700, 2000, 2200, 1800, 2400, 2600, 1900, 2300, 2500, 2100],
            'matches_played': [30, 25, 28, 22, 27, 29, 21, 26, 28, 24],  # All above MIN_PLAYER_MATCHES=3
            'goals_per_90': [0.7, 0.5, 0.3, 0.2, 0.1, 0.9, 0.4, 0.4, 0.1, 0.05],
            'assists_per_90': [0.3, 0.2, 0.4, 0.3, 0.2, 0.2, 0.5, 0.3, 0.2, 0.1],
            'position': ['F', 'F', 'F', 'M', 'M', 'F', 'M', 'M', 'M', 'D'],
        })
        cache_path = tmp_path / 'player_stats.parquet'
        player_cache.to_parquet(cache_path)
        return ExternalFeatureInjector(
            referee_cache_path=str(tmp_path / 'referee_empty.parquet'),
            player_stats_cache_path=str(cache_path),
            enable_referee=False,
            enable_weather=False,
            enable_lineups=True,
        )

    def test_inject_lineup_features_with_known_players(self, injector):
        """Known players should inject actual stats."""
        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {
            'home_lineup': {
                'starting_xi': [
                    {'id': 101, 'name': 'Salah'},
                    {'id': 102, 'name': 'Nunez'},
                    {'id': 103, 'name': 'Diaz'},
                    {'id': 104, 'name': 'Szoboszlai'},
                    {'id': 105, 'name': 'Mac Allister'},
                ]
            },
            'away_lineup': {
                'starting_xi': [
                    {'id': 201, 'name': 'Haaland'},
                    {'id': 202, 'name': 'De Bruyne'},
                    {'id': 203, 'name': 'Foden'},
                    {'id': 204, 'name': 'Rodri'},
                    {'id': 205, 'name': 'Stones'},
                ]
            },
        }

        result = injector._inject_lineup_features(features_df.copy(), match_info)

        # Home XI avg rating = (7.8+7.2+7.0+7.1+7.3)/5 = 7.28
        assert result['home_xi_avg_rating'].iloc[0] == pytest.approx(7.28, rel=0.01)
        # Away XI avg rating = (8.2+7.9+7.4+7.6+7.0)/5 = 7.62
        assert result['away_xi_avg_rating'].iloc[0] == pytest.approx(7.62, rel=0.01)
        # Rating diff = 7.28 - 7.62 = -0.34
        assert result['lineup_rating_diff'].iloc[0] == pytest.approx(-0.34, rel=0.1)

    def test_inject_lineup_features_goals_assists(self, injector):
        """Goals and assists per 90 should sum across lineup."""
        features_df = pd.DataFrame({'fixture_id': [1]})
        match_info = {
            'home_lineup': {
                'starting_xi': [{'id': 101}, {'id': 102}]  # Salah + Nunez
            },
            'away_lineup': {
                'starting_xi': [{'id': 201}, {'id': 202}]  # Haaland + De Bruyne
            },
        }

        result = injector._inject_lineup_features(features_df.copy(), match_info)

        # Home goals_per_90 = 0.7 + 0.5 = 1.2
        assert result['home_xi_goals_per_90'].iloc[0] == pytest.approx(1.2, rel=0.01)
        # Home assists_per_90 = 0.3 + 0.2 = 0.5
        assert result['home_xi_assists_per_90'].iloc[0] == pytest.approx(0.5, rel=0.01)
        # Away goals_per_90 = 0.9 + 0.4 = 1.3
        assert result['away_xi_goals_per_90'].iloc[0] == pytest.approx(1.3, rel=0.01)

    def test_inject_lineup_no_lineup_keeps_existing(self, injector):
        """Missing lineups should keep existing features unchanged (no bias)."""
        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_xi_avg_rating': [7.4],  # Historical team average
            'away_xi_avg_rating': [7.1],
        })
        match_info = {}  # No lineups

        result = injector._inject_lineup_features(features_df.copy(), match_info)

        # Should NOT override with defaults - keep historical values
        assert result['home_xi_avg_rating'].iloc[0] == 7.4
        assert result['away_xi_avg_rating'].iloc[0] == 7.1

    def test_inject_lineup_partial_lineup_keeps_existing(self, injector):
        """Only one lineup provided should keep existing features."""
        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_xi_avg_rating': [7.2],
        })
        match_info = {
            'home_lineup': {'starting_xi': [{'id': 101}]},
            # away_lineup missing
        }

        result = injector._inject_lineup_features(features_df.copy(), match_info)

        # Should NOT override - keep historical
        assert result['home_xi_avg_rating'].iloc[0] == 7.2

    def test_inject_lineup_unknown_players_keeps_existing(self, injector):
        """Lineup with all unknown player IDs should keep existing features."""
        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_xi_avg_rating': [7.3],
            'away_xi_avg_rating': [6.9],
        })
        match_info = {
            'home_lineup': {'starting_xi': [{'id': 999}, {'id': 998}]},
            'away_lineup': {'starting_xi': [{'id': 997}, {'id': 996}]},
        }

        result = injector._inject_lineup_features(features_df.copy(), match_info)

        # All players unknown - should NOT override with defaults
        # Keep historical team averages
        assert result['home_xi_avg_rating'].iloc[0] == 7.3
        assert result['away_xi_avg_rating'].iloc[0] == 6.9

    def test_extract_player_ids_direct_format(self, injector):
        """Test player ID extraction from direct format."""
        lineup = {'starting_xi': [{'id': 101}, {'id': 102}, {'id': 103}]}
        ids = injector._extract_player_ids(lineup)
        assert ids == [101, 102, 103]

    def test_extract_player_ids_nested_format(self, injector):
        """Test player ID extraction from API-Football nested format."""
        lineup = {
            'starting_xi': [
                {'player': {'id': 101, 'name': 'Player A'}},
                {'player': {'id': 102, 'name': 'Player B'}},
            ]
        }
        ids = injector._extract_player_ids(lineup)
        assert ids == [101, 102]

    def test_extract_player_ids_list_format(self, injector):
        """Test player ID extraction from direct list."""
        lineup = [{'id': 101}, {'id': 102}]
        ids = injector._extract_player_ids(lineup)
        assert ids == [101, 102]

    def test_extract_player_ids_startXI_key(self, injector):
        """Test player ID extraction with startXI key (API-Football format)."""
        lineup = {'startXI': [{'id': 101}, {'id': 102}]}
        ids = injector._extract_player_ids(lineup)
        assert ids == [101, 102]


class TestFullInjectionWithLineups:
    """Test complete injection workflow including lineups."""

    @pytest.fixture
    def full_injector(self, tmp_path):
        """Create injector with all data sources."""
        from src.ml.feature_injector import ExternalFeatureInjector

        # Referee cache
        referee_cache = pd.DataFrame({
            'referee_name': ['Michael Oliver'],
            'matches': [50],
            'total_yellows': [200],
            'total_reds': [10],
            'total_fouls': [1100],
            'total_corners': [520],
            'home_wins': [23],
            'draws': [12],
            'away_wins': [15],
            'total_goals': [140],
        })
        ref_path = tmp_path / 'referee_stats.parquet'
        referee_cache.to_parquet(ref_path)

        # Player cache
        player_cache = pd.DataFrame({
            'player_id': [101, 201],
            'player_name': ['Salah', 'Haaland'],
            'avg_rating': [7.8, 8.2],
            'total_minutes': [2700, 2600],
            'matches_played': [30, 29],
            'goals_per_90': [0.7, 0.9],
            'assists_per_90': [0.3, 0.2],
            'position': ['F', 'F'],
        })
        player_path = tmp_path / 'player_stats.parquet'
        player_cache.to_parquet(player_path)

        return ExternalFeatureInjector(
            referee_cache_path=str(ref_path),
            player_stats_cache_path=str(player_path),
            enable_referee=True,
            enable_weather=True,
            enable_lineups=True,
        )

    def test_full_injection_all_features(self, full_injector):
        """Full injection should add referee, weather, AND lineup features."""
        # Mock weather
        mock_collector = Mock()
        mock_collector.fetch_forecast.return_value = {
            'temperature': 12.0,
            'precipitation': 0.0,
            'wind_speed': 10.0,
            'humidity': 70,
            'weather_code': 0,
        }
        full_injector.weather_collector = mock_collector

        features_df = pd.DataFrame({
            'fixture_id': [12345],
            'home_elo': [1600],
        })

        match_info = {
            'referee': 'Michael Oliver',
            'venue_city': 'London',
            'kickoff': datetime(2026, 2, 8, 15, 0),
            'home_lineup': {'starting_xi': [{'id': 101}]},
            'away_lineup': {'starting_xi': [{'id': 201}]},
        }

        result = full_injector.inject_features(features_df, match_info)

        # Original preserved
        assert result['home_elo'].iloc[0] == 1600

        # Referee features
        assert 'ref_cards_avg' in result.columns
        assert result['ref_matches'].iloc[0] == 50

        # Weather features
        assert 'weather_temp' in result.columns
        assert result['weather_temp'].iloc[0] == 12.0

        # Lineup features
        assert 'home_xi_avg_rating' in result.columns
        assert result['home_xi_avg_rating'].iloc[0] == pytest.approx(7.8, rel=0.01)
        assert result['away_xi_avg_rating'].iloc[0] == pytest.approx(8.2, rel=0.01)
        assert result['lineup_rating_diff'].iloc[0] == pytest.approx(-0.4, rel=0.1)
