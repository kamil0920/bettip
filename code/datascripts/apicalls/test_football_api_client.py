import unittest
from unittest.mock import Mock, patch
import json
import time
from pathlib import Path
import tempfile
import os
from datetime import date

import requests

from .api_call import FootballAPIClient, APIError, TokenBucket


class TestTokenBucket(unittest.TestCase):
    def test_token_consumption(self):
        """Test that token bucket properly limits requests."""
        bucket = TokenBucket(rate_per_min=60, burst=10)  # 1 per second, burst of 10

        # Should be able to consume 10 tokens immediately
        start_time = time.time()
        for i in range(10):
            bucket.consume(1.0)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 0.1)  # Should be nearly instant

        # 11th token should take some time
        start_time = time.time()
        bucket.consume(1.0)
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.5)  # Should take at least 0.5s

    def test_token_refill(self):
        """Test that tokens are refilled over time."""
        bucket = TokenBucket(rate_per_min=60, burst=2)  # 1 per second, burst of 2

        # Consume all tokens
        bucket.consume(2.0)

        # Wait a bit for refill
        time.sleep(1.1)

        # Should be able to consume again
        start_time = time.time()
        bucket.consume(1.0)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 0.1)


class TestFootballAPIClient(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock environment variables
        self.env_vars = {
            'API_FOOTBALL_KEY': 'test_key_123',
            'API_BASE_URL': 'https://test.api.com',
            'DAILY_LIMIT': '100',
            'PER_MIN_LIMIT': '10',
            'STATE_PATH': os.path.join(self.temp_dir, 'test_state.json')
        }

        # Apply environment patches
        self.env_patcher = patch.dict(os.environ, self.env_vars)
        self.env_patcher.start()

        # Reload module constants after patching env vars
        import importlib
        from . import api_call
        importlib.reload(api_call)

        globals()['TokenBucket'] = api_call.TokenBucket
        globals()['FootballAPIClient'] = api_call.FootballAPIClient
        globals()['APIError'] = api_call.APIError

    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test client initialization with env variables."""
        client = FootballAPIClient()

        self.assertEqual(client.api_key, 'test_key_123')
        self.assertEqual(client.base_url, 'https://test.api.com')
        self.assertEqual(client.daily_limit, 100)
        self.assertEqual(client.per_min_limit, 10)
        self.assertIsInstance(client.bucket, TokenBucket)

        def test_state_file_path(self):
            """Test that state file path is set correctly."""
            client = FootballAPIClient()
            expected_path = Path(os.path.join(self.temp_dir, 'test_state.json'))
            self.assertEqual(client.state_path, expected_path)

        @patch('requests.Session.get')
        def test_successful_request(self, mock_get):
            """Test successful API request."""
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'response': [{'id': 1, 'name': 'test'}]}
            mock_get.return_value = mock_response

            client = FootballAPIClient()
            result = client._make_request('/test', {'param': 'value'})

            self.assertEqual(result, {'response': [{'id': 1, 'name': 'test'}]})
            mock_get.assert_called_once()

            # Check that request was recorded
            self.assertEqual(client.state['count'], 1)

        @patch('requests.Session.get')
        def test_retry_on_server_error(self, mock_get):
            """Test retry logic for server errors."""
            # Mock server error then success
            error_response = Mock()
            error_response.status_code = 503
            error_response.headers = {}

            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {'response': 'success'}

            mock_get.side_effect = [error_response, success_response]

            client = FootballAPIClient()
            with patch('time.sleep'):  # Speed up test by mocking sleep
                result = client._make_request('/test', {})

            self.assertEqual(result, {'response': 'success'})
            self.assertEqual(mock_get.call_count, 2)

        @patch('requests.Session.get')
        def test_retry_after_header(self, mock_get):
            """Test that Retry-After header is respected."""
            error_response = Mock()
            error_response.status_code = 429
            error_response.headers = {'Retry-After': '2'}

            success_response = Mock()
            success_response.status_code = 200
            success_response.json.return_value = {'response': 'success'}

            mock_get.side_effect = [error_response, success_response]

            client = FootballAPIClient()

            with patch('time.sleep') as mock_sleep:
                result = client._make_request('/test', {})

            # Should have slept for 2 seconds as specified in header
            mock_sleep.assert_called_with(2)
            self.assertEqual(result, {'response': 'success'})

        def test_daily_limit_enforcement(self):
            """Test daily limit is enforced."""
            client = FootballAPIClient()

            # Manually set state to near limit
            client.state = {"date": str(date.today()), "count": 99}
            client.daily_limit = 100

            # This should work (request #100 will be allowed)
            client._check_daily_limit()

            # This should fail (already at limit)
            client.state["count"] = 100
            with self.assertRaises(APIError) as context:
                client._check_daily_limit()

            self.assertIn("Daily limit 100 reached", str(context.exception))

        def test_state_persistence(self):
            """Test that API state is persisted correctly."""
            # First client
            client1 = FootballAPIClient()
            initial_count = client1.state.get("count", 0)

            # Record some requests
            client1._record_request()
            client1._record_request()

            self.assertEqual(client1.state["count"], initial_count + 2)

            # Create new client instance - should load previous state
            client2 = FootballAPIClient()
            self.assertEqual(client2.state["count"], initial_count + 2)

        def test_state_reset_new_day(self):
            """Test that state resets for new day."""
            client = FootballAPIClient()

            # Create state file with old date
            old_state = {"date": "2023-01-01", "count": 50}
            with open(client.state_path, 'w') as f:
                json.dump(old_state, f)

            # Load state - should reset because date is old
            new_state = client._load_state()
            self.assertEqual(new_state["count"], 0)
            self.assertEqual(new_state["date"], str(date.today()))

    class TestIntegration(unittest.TestCase):
        """Integration tests with actual methods."""

        def setUp(self):
            self.temp_dir = tempfile.mkdtemp()

            self.env_patcher = patch.dict(os.environ, {
                'API_FOOTBALL_KEY': 'test_key_123',
                'API_BASE_URL': 'https://test.api.com',
                'DAILY_LIMIT': '100',
                'PER_MIN_LIMIT': '10',
                'STATE_PATH': os.path.join(self.temp_dir, 'test_state.json')
            })
            self.env_patcher.start()

            # Reload module
            import importlib
            from . import api_call
            importlib.reload(api_call)

        def tearDown(self):
            self.env_patcher.stop()
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        @patch('requests.Session.get')
        def test_get_fixtures(self, mock_get):
            """Test get_fixtures method."""
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': [
                    {'fixture': {'id': 1, 'date': '2023-01-01'}, 'teams': {}},
                    {'fixture': {'id': 2, 'date': '2023-01-02'}, 'teams': {}}
                ]
            }
            mock_get.return_value = mock_response

            client = FootballAPIClient()
            fixtures = client.get_fixtures(league_id=39, season=2023)

            self.assertEqual(len(fixtures), 2)
            self.assertEqual(fixtures[0]['fixture']['id'], 1)

            # Verify correct endpoint was called
            call_args = mock_get.call_args
            self.assertIn('/fixtures', call_args[0][0])
            self.assertEqual(call_args[1]['params'], {'league': 39, 'season': 2023})

        @patch('requests.Session.get')
        def test_get_player_statistics(self, mock_get):
            """Test get_player_statistics method."""
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': [
                    {'player': {'id': 1, 'name': 'Player 1'}, 'statistics': [{}]},
                    {'player': {'id': 2, 'name': 'Player 2'}, 'statistics': [{}]}
                ]
            }
            mock_get.return_value = mock_response

            client = FootballAPIClient()
            players = client.get_player_statistics(fixture_id=123)

            self.assertEqual(len(players), 2)
            self.assertEqual(players[0]['player']['name'], 'Player 1')

            # Verify correct endpoint was called
            call_args = mock_get.call_args
            self.assertIn('/players', call_args[0][0])
            self.assertEqual(call_args[1]['params'], {'fixture': 123})

        @patch('requests.Session.get')
        def test_api_error_handling(self, mock_get):
            """Test API error handling in high-level methods."""
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("Not Found")
            mock_get.return_value = mock_response

            client = FootballAPIClient()

            with self.assertRaises(APIError) as context:
                client.get_fixtures(league_id=999, season=2023)

            self.assertIn("Failed to get fixtures", str(context.exception))

    if __name__ == '__main__':
        # Run with verbose output
        unittest.main(verbosity=2)
