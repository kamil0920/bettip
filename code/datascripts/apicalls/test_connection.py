#!/usr/bin/env python3
"""
Quick connection test and API status check.
"""

from .api_call import FootballAPIClient, APIError


def test_connection():
    """Test API connection and show status."""
    print("🔍 Testing API connection...")

    client = FootballAPIClient()

    # Check configuration
    print(f"🔑 API Key: {'✅ Set' if client.api_key else '❌ Missing'}")
    print(f"🌐 Base URL: {client.base_url}")
    print(f"📊 Daily Limit: {client.daily_limit}")
    print(f"⏱️  Rate Limit: {client.per_min_limit}/min")
    print(f"📈 Used Today: {client.state.get('count', 0)}")

    if not client.api_key:
        print("\n❌ Please set API_FOOTBALL_KEY in your .env file")
        return False

    try:
        # Test simple endpoint
        print("\n🧪 Testing /status endpoint...")
        response = client._make_request('/status', {})
        print(f"✅ Connection successful!")

        # Show API info from response
        if 'response' in response:
            api_info = response['response']
            print(f"📊 API Info:")
            for key, value in api_info.items():
                print(f"   {key}: {value}")

        return True

    except APIError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    test_connection()
