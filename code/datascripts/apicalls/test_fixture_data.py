#!/usr/bin/env python3
"""
Test API endpoints based on official documentation patterns
"""

from api_call import FootballAPIClient
import json


def test_fixture_detailed_data():
    """Test fixture-related endpoints that might contain player info."""
    client = FootballAPIClient()
    fixture_id = 1035037

    print(f"🔍 Testing fixture detailed data for ID: {fixture_id}")

    # Test fixture-related endpoints that often contain player data
    fixture_endpoints = [
        ('/fixtures/lineups', {'fixture': fixture_id}),
        ('/fixtures/events', {'fixture': fixture_id}),
        ('/fixtures/statistics', {'fixture': fixture_id}),
        ('/fixtures/players', {'fixture': fixture_id}),  # Alternative endpoint
    ]

    for endpoint, params in fixture_endpoints:
        print(f"\n📡 Testing: {endpoint}")

        try:
            response = client._make_request(endpoint, params)
            data = response.get('response', [])

            if data and len(data) > 0:
                print(f"   ✅ SUCCESS! Found {len(data)} records")
                print(f"   Keys: {list(data[0].keys()) if data[0] else 'Empty'}")

                # Save successful response
                filename = f"success_{endpoint.replace('/', '_')}_{fixture_id}.json"
                with open(filename, 'w') as f:
                    json.dump(response, f, indent=2, default=str)

                print(f"   💾 Saved to: {filename}")

                # If this is lineups, show structure
                if 'lineups' in endpoint and data:
                    print(f"   📋 Lineup structure:")
                    for i, team_lineup in enumerate(data):
                        if 'team' in team_lineup:
                            team_name = team_lineup['team'].get('name', f'Team {i + 1}')
                            print(f"     Team: {team_name}")

                            if 'startXI' in team_lineup:
                                print(f"     Starting XI: {len(team_lineup['startXI'])} players")
                            if 'substitutes' in team_lineup:
                                print(f"     Substitutes: {len(team_lineup['substitutes'])} players")

                return True
            else:
                print(f"   ❌ Empty response")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False


def main():
    print("🔧 FIXTURE DATA DISCOVERY")
    print("=" * 40)

    success = test_fixture_detailed_data()

    if success:
        print(f"\n✅ Found working endpoint!")
    else:
        print(f"\n❌ No fixture data endpoints working")
        print(f"💡 Your API plan might not include detailed match data")


if __name__ == "__main__":
    main()
