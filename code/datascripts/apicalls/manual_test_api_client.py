#!/usr/bin/env python3
"""
Manual testing script for FootballAPIClient without ModelConfig.
"""

import logging
import time
import os
from pathlib import Path
from unittest.mock import patch

from api_call import FootballAPIClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_initialization():
    """Test basic client initialization."""
    print("Testing client initialization...")

    try:
        client = FootballAPIClient()
        print(f"✓ Client initialized successfully")
        print(f"  API Key: {client.api_key[:10]}..." if client.api_key else "  API Key: None")
        print(f"  Base URL: {client.base_url}")
        print(f"  Daily Limit: {client.daily_limit}")
        print(f"  Per Minute Limit: {client.per_min_limit}")
        print(f"  State Path: {client.state_path}")
        print(f"  Current State: {client.state}")
        return client
    except Exception as e:
        print(f"✗ Client initialization failed: {e}")
        return None


def test_rate_limiting():
    """Test rate limiting behavior with mock requests."""
    print("\nTesting rate limiting...")

    # Test with temporary env vars for faster testing
    with patch.dict(os.environ, {
        'PER_MIN_LIMIT': '6',  # 6 requests per minute = 1 per 10 seconds
        'STATE_PATH': 'test_rate_limit_state.json'
    }):
        # Need to reimport to get new env values
        import importlib
        import api_call
        importlib.reload(api_call)

        client = api_call.FootballAPIClient()

        print(f"Testing with rate limit: {client.per_min_limit} requests/minute")

        start_time = time.time()
        for i in range(3):
            print(f"Making request {i + 1}...")
            request_start = time.time()

            # Just test the rate limiting, not actual API call
            client.bucket.consume(1.0)

            elapsed = time.time() - request_start
            print(f"  Request {i + 1} took {elapsed:.3f}s")

        total_elapsed = time.time() - start_time
        print(f"Total time for 3 requests: {total_elapsed:.3f}s")

    # Clean up test file
    test_file = Path('test_rate_limit_state.json')
    if test_file.exists():
        test_file.unlink()


def test_state_persistence():
    """Test that state persists across client instances."""
    print("\nTesting state persistence...")

    test_state_path = 'test_persistence_state.json'

    with patch.dict(os.environ, {'STATE_PATH': test_state_path}):
        # Reload to get new STATE_PATH
        import importlib
        import api_call
        importlib.reload(api_call)

        # First client
        client1 = api_call.FootballAPIClient()
        initial_count = client1.state.get("count", 0)
        print(f"Initial request count: {initial_count}")

        # Simulate some requests
        client1._record_request()
        client1._record_request()
        client1._record_request()
        print(f"After 3 requests, count: {client1.state['count']}")

        # Second client should load the same state
        client2 = api_call.FootballAPIClient()
        print(f"New client loaded count: {client2.state['count']}")

        if client2.state['count'] == client1.state['count']:
            print("✓ State persistence works correctly")
        else:
            print("✗ State persistence failed")

    # Clean up test file
    test_file = Path(test_state_path)
    if test_file.exists():
        test_file.unlink()


def test_daily_limit():
    """Test daily limit enforcement."""
    print("\nTesting daily limit enforcement...")

    with patch.dict(os.environ, {
        'DAILY_LIMIT': '5',
        'STATE_PATH': 'test_daily_limit_state.json'
    }):
        import importlib
        import api_call
        importlib.reload(api_call)

        from api_call import APIError

        client = api_call.FootballAPIClient()
        print(f"Testing with daily limit: {client.daily_limit}")

        # Simulate reaching the limit
        client.state['count'] = 4
        client._save_state()

        print(f"Current count: {client.state['count']}")

        # This should work (request #5)
        try:
            client._check_daily_limit()
            print("✓ Request #5 allowed")
        except APIError as e:
            print(f"✗ Request #5 unexpectedly blocked: {e}")

        # Set to limit
        client.state['count'] = 5

        # This should fail (request #6)
        try:
            client._check_daily_limit()
            print("✗ Request #6 should have been blocked")
        except APIError as e:
            print(f"✓ Request #6 correctly blocked: {e}")

    # Clean up test file
    test_file = Path('test_daily_limit_state.json')
    if test_file.exists():
        test_file.unlink()


def test_with_real_api():
    """Test with real API calls (use carefully!)."""
    print("\nTesting with real API calls...")
    print("WARNING: This will use your API quota!")

    response = input("Do you want to proceed with real API calls? (y/N): ")
    if response.lower() != 'y':
        print("Skipping real API tests")
        return

    client = FootballAPIClient()

    try:
        print("Testing /status endpoint...")
        response = client._make_request('/status', {})
        print(f"✓ Status response: {response}")

        print("Testing /leagues endpoint...")
        response = client._make_request('/leagues', {'country': 'England'})
        leagues = response.get('response', [])
        print(f"✓ Found {len(leagues)} leagues")

        if leagues:
            premier_league = next((l for l in leagues if 'Premier' in l['league']['name']), None)
            if premier_league:
                print(f"✓ Found Premier League: {premier_league['league']['name']}")

    except Exception as e:
        print(f"✗ Real API test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("FootballAPIClient Manual Testing (No ModelConfig)")
    print("=" * 60)

    # Test 1: Initialization
    client = test_initialization()
    if not client:
        print("Cannot continue - initialization failed")
        exit(1)

    # Test 2: Rate limiting
    test_rate_limiting()

    # Test 3: State persistence
    test_state_persistence()

    # Test 4: Daily limit
    test_daily_limit()

    # Test 5: Real API (optional)
    test_with_real_api()

    print("\n" + "=" * 60)
    print("Testing completed!")
