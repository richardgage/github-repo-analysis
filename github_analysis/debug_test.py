#!/usr/bin/env python3
"""
debug_test.py
Simple test to debug what's going wrong
"""

import os
from dotenv import load_dotenv
from github_collector import GitHubDataCollector

def main():
    print("🔍 DEBUG TEST - Testing individual components")
    print("=" * 50)
    
    # Load token
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    print(f"✓ Token loaded: {token[:10]}...")
    
    # Create collector
    collector = GitHubDataCollector(token)
    print("✓ Collector created")
    
    # Test with React
    owner, repo = 'facebook', 'react'
    print(f"\n🧪 Testing {owner}/{repo}")
    
    print("1. Testing contributors...")
    try:
        print("   Calling get_contributors_data...")
        contributors = collector.get_contributors_data(owner, repo)
        print(f"   Returned value type: {type(contributors)}")
        print(f"   Returned value: {contributors}")
        
        if contributors is None:
            print("   ERROR: Function returned None!")
        elif isinstance(contributors, list):
            print(f"   SUCCESS: Got {len(contributors)} contributors")
            if contributors:
                print(f"   First contributor: {contributors[0].get('login', 'Unknown')}")
        else:
            print(f"   ERROR: Unexpected type: {type(contributors)}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("2. Testing releases...")
    try:
        releases = collector.get_releases_data(owner, repo)
        print(f"   SUCCESS: Got {len(releases)} releases")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("3. Testing pull requests...")
    try:
        prs = collector.get_pull_requests_data(owner, repo)
        print(f"   SUCCESS: Got {len(prs)} pull requests")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("4. Testing repository contents...")
    try:
        contents = collector.get_repository_contents(owner, repo)
        print(f"   SUCCESS: Got {len(contents)} files")
        if contents:
            print(f"   First file: {contents[0].get('name', 'Unknown')}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n✅ Debug test completed!")

if __name__ == "__main__":
    main()