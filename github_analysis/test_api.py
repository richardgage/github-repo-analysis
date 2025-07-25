#!/usr/bin/env python3
"""
test_api.py
Simple test script to verify GitHub API setup

This script tests your GitHub token and API access before running
the full data collection analysis.
"""

import requests
import os
from dotenv import load_dotenv

def test_basic_connection():
    """Test if we can connect to GitHub API"""
    print("Testing GitHub API connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get token from environment
    token = os.getenv('GITHUB_TOKEN')
    
    if not token:
        print("ERROR: No GitHub token found!")
        print("Make sure you have a .env file with GITHUB_TOKEN=your_token")
        return False
    
    print(f"✓ Token found: {token[:10]}...")
    
    # Set up headers
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Test rate limit endpoint
    print("\nTesting API access...")
    response = requests.get("https://api.github.com/rate_limit", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        remaining = data['rate']['remaining']
        limit = data['rate']['limit']
        print(f"✓ API access successful!")
        print(f"✓ Rate limit: {remaining}/{limit} calls remaining")
        return True
    else:
        print(f"API access failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_repository_access():
    """Test accessing a simple repository"""
    print("\n" + "="*50)
    print("Testing repository data access...")
    
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Test with a well-known repository
    test_repo = "octocat/Hello-World"
    url = f"https://api.github.com/repos/{test_repo}"
    
    print(f"Fetching data for {test_repo}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Repository data retrieved successfully!")
        print(f"  Name: {data['full_name']}")
        print(f"  Stars: {data['stargazers_count']}")
        print(f"  Language: {data['language']}")
        print(f"  Open Issues: {data['open_issues_count']}")
        print(f"  Created: {data['created_at']}")
        return True
    else:
        print(f"Failed to get repository data: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_issues_access():
    """Test accessing issues data"""
    print("\n" + "="*50)
    print("Testing issues data access...")
    
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Test with a repository that has issues
    test_repo = "octocat/Hello-World"
    url = f"https://api.github.com/repos/{test_repo}/issues"
    params = {'state': 'all', 'per_page': 5}  # Just get 5 issues for testing
    
    print(f"Fetching issues for {test_repo}...")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        issues = response.json()
        print(f"✓ Issues data retrieved successfully!")
        print(f"  Found {len(issues)} issues")
        
        if issues:
            first_issue = issues[0]
            print(f"  First issue: #{first_issue['number']} - {first_issue['title']}")
            print(f"  State: {first_issue['state']}")
            print(f"  Comments: {first_issue['comments']}")
        
        return True
    else:
        print(f"Failed to get issues data: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    """Run all tests"""
    print("GitHub API Setup Test")
    print("="*50)
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("\nBasic connection failed. Check your token setup.")
        return
    
    # Test 2: Repository access
    if not test_repository_access():
        print("\nRepository access failed.")
        return
    
    # Test 3: Issues access
    if not test_issues_access():
        print("\nIssues access failed.")
        return
    
    print("\n" + "="*50)
    print("All tests passed! Your GitHub API setup is working!")
    print("You can now run the full data collection script.")

if __name__ == "__main__":
    main()