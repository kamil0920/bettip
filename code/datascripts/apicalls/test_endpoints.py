#!/usr/bin/env python3
"""
Test different API endpoints and parameters for player data
"""

from .api_call import FootballAPIClient
import json


def test_player_endpoints():
    """Test different endpoints that might contain player data."""
    client = FootballAPIClient()
    fixture_id = 1035037  # Burnley vs Manchester City from 2023

    print("🔍 Testing different player-related endpoints...")

    endpoints_to_test = [
        ('/players', {'fixture': fixture_id}),
        ('/players/squads', {'team': 17}),  # Manchester City team ID
        ('/players/squads', {'team': 44}),  # Burnley team ID
        ('/fixtures/players', {'fixture': fixture_id}),
        ('/fixtures/statistics', {'fixture': fixture_id}),
        ('/fixtures/lineups', {'fixture': fixture_id}),
        ('/fixtures/events', {'fixture': fixture_id}),

        ('/players', {'league': 39, 'season': 2023}),
        ('/players/topscorers', {'league': 39, 'season': 2023}),
        ('/players/topassists', {'league': 39, 'season': 2023}),
    ]

    results = {}

    for endpoint, params in endpoints_to_test:
        print(f"\n🧪 Testing: {endpoint} with {params}")

        try:
            response = client._make_request(endpoint, params)
            response_data = response.get('response', [])

            print(f"   Status: ✅ Success")
            print(f"   Results: {len(response_data) if hasattr(response_data, '__len__') else 'No length'}")

            if response_data and len(response_data) > 0:
                print(f"   🎉 FOUND DATA!")
                print(
                    f"   First item keys: {list(response_data[0].keys()) if isinstance(response_data, list) and response_data[0] else 'N/A'}")

                with open(f'successful_{endpoint.replace("/", "_")}_{fixture_id}.json', 'w') as f:
                    json.dump({
                        'endpoint': endpoint,
                        'params': params,
                        'response': response
                    }, f, indent=2, default=str)

                results[endpoint] = True
            else:
                print(f"   Empty response")
                results[endpoint] = False

        except Exception as e:
            print(f"   Status: ❌ Error - {e}")
            results[endpoint] = False

    print(f"\n📊 SUMMARY:")
    successful = [ep for ep, success in results.items() if success]
    if successful:
        print(f"✅ Working endpoints: {successful}")
    else:
        print(f"❌ No working endpoints found")

    return successful


def test_team_based_queries():
    """Test team-based player queries instead of fixture-based."""
    client = FootballAPIClient()

    print(f"\n🔍 Testing team-based player queries...")

    premier_league_teams = [
        (33, "Manchester United"),
        (40, "Liverpool"),
        (42, "Arsenal"),
        (50, "Manchester City"),
        (49, "Chelsea")
    ]

    for team_id, team_name in premier_league_teams[:2]:
        print(f"\n🧪 Testing team: {team_name} (ID: {team_id})")

        try:
            # Try different team-based endpoints
            endpoints = [
                ('/players', {'team': team_id, 'season': 2023}),
                ('/players/squads', {'team': team_id}),
            ]

            for endpoint, params in endpoints:
                print(f"   Trying: {endpoint}")
                response = client._make_request(endpoint, params)
                response_data = response.get('response', [])

                if response_data and len(response_data) > 0:
                    print(f"   ✅ Success: {len(response_data)} records")

                    # Save first successful team query
                    with open(f'team_players_{team_name}_{team_id}.json', 'w') as f:
                        json.dump(response, f, indent=2, default=str)

                    return True
                else:
                    print(f"   ❌ Empty response")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False


def check_available_endpoints():
    """Check what endpoints are actually available in your API plan."""
    client = FootballAPIClient()

    print(f"\n🔍 Checking API plan capabilities...")

    try:
        # Get status to see plan details
        status_response = client._make_request('/status', {})

        print(f"📋 Account details:")
        account = status_response.get('response', {}).get('account', {})
        subscription = status_response.get('response', {}).get('subscription', {})

        print(f"   Plan: {subscription.get('plan', 'Unknown')}")
        print(f"   Active: {subscription.get('active', 'Unknown')}")
        print(f"   End date: {subscription.get('end', 'Unknown')}")

        # Test some basic endpoints to see what's available
        basic_endpoints = [
            '/leagues',
            '/teams',
            '/fixtures',
            '/standings'
        ]

        print(f"\n📋 Testing basic endpoint availability:")
        for endpoint in basic_endpoints:
            try:
                response = client._make_request(endpoint, {'league': 39, 'season': 2023})
                print(f"   {endpoint}: ✅ Available")
            except Exception as e:
                print(f"   {endpoint}: ❌ Error - {e}")

    except Exception as e:
        print(f"❌ Error checking API status: {e}")


def main():
    print("🔧 API ENDPOINT DISCOVERY")
    print("=" * 50)

    # Step 1: Check plan and basic endpoints
    check_available_endpoints()

    # Step 2: Test player-related endpoints
    successful_endpoints = test_player_endpoints()

    # Step 3: Test team-based queries if fixture-based don't work
    if not successful_endpoints:
        print(f"\n🔄 Fixture-based queries failed, trying team-based...")
        team_success = test_team_based_queries()

        if not team_success:
            print(f"\n❌ No player data endpoints seem to work")
            print(f"💡 Possible solutions:")
            print(f"   1. Player data might not be included in your API plan")
            print(f"   2. Different endpoint structure might be needed")
            print(f"   3. Different parameters might be required")
            print(f"   4. Player data might be in different endpoints (lineups, events)")
