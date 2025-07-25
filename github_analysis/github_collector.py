#!/usr/bin/env python3
"""
github_collector.py
GitHub API Data Collection Module

This module contains the GitHubDataCollector class for fetching repository data
from the GitHub API with proper rate limiting and error handling.
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GitHubDataCollector:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None
    
    def check_rate_limit(self):
        """Check and handle GitHub API rate limits"""
        response = requests.get(f"{self.base_url}/rate_limit", headers=self.headers)
        if response.status_code == 200:
            rate_data = response.json()
            self.rate_limit_remaining = rate_data['rate']['remaining']
            self.rate_limit_reset = rate_data['rate']['reset']
            
            if self.rate_limit_remaining < 100:
                reset_time = datetime.fromtimestamp(self.rate_limit_reset)
                wait_time = (reset_time - datetime.now()).total_seconds()
                print(f"Rate limit low. Waiting {wait_time} seconds...")
                time.sleep(wait_time + 10)
    
    def make_request(self, url, params=None):
        """Make API request with rate limiting"""
        self.check_rate_limit()
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 403 and 'rate limit' in response.text.lower():
            print("Rate limit exceeded. Waiting...")
            time.sleep(3600)  # Wait 1 hour
            response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    
    def get_repository_data(self, owner, repo):
        """Get basic repository information"""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        return self.make_request(url)
    
    def get_issues_data(self, owner, repo, state='all', per_page=100, months_back=6):
        """Get issues data with optional time filtering"""
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {
            'state': state,
            'per_page': per_page,
            'sort': 'created',
            'direction': 'desc'
        }
        
        # Add time filter for recent issues
        if months_back:
            since_date = datetime.now() - timedelta(days=months_back * 30)
            params['since'] = since_date.isoformat()
        
        all_issues = []
        page = 1
        
        while len(all_issues) < 500:  # Reasonable limit
            params['page'] = page
            issues = self.make_request(url, params)
            
            if not issues or len(issues) == 0:
                break
                
            # Filter out pull requests (GitHub treats PRs as issues)
            issues = [issue for issue in issues if 'pull_request' not in issue]
            all_issues.extend(issues)
            
            if len(issues) < per_page:
                break
            
            page += 1
            time.sleep(0.5)  # Be nice to the API
        
        return all_issues
    
    def get_contributors_data(self, owner, repo):
        """Get contributors information"""
        url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
        params = {'per_page': 100}
        
        all_contributors = []
        page = 1
        
        while True:
            params['page'] = page
            contributors = self.make_request(url, params)
            
            if not contributors or len(contributors) == 0:
                break
                
            all_contributors.extend(contributors)
            
            if len(contributors) < 100:
                break
            
            page += 1
            time.sleep(0.5)
        
        return all_contributors
    
    def get_commits_data(self, owner, repo, since_date=None):
        """Get commits data within a specific time window"""
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {'per_page': 100}
        
        # Always use a since_date for consistent time windows
        if not since_date:
            since_date = datetime.now() - timedelta(days=180)  # Default: 6 months
        
        params['since'] = since_date.isoformat()
        
        all_commits = []
        page = 1
        
        # Remove arbitrary commit limit, let time window control it
        while page <= 20:  # Max 20 pages (2000 commits) to prevent runaway
            params['page'] = page
            commits = self.make_request(url, params)
            
            if not commits or len(commits) == 0:
                break
                
            all_commits.extend(commits)
            
            if len(commits) < 100:
                break
            
            page += 1
            time.sleep(0.5)
        
        return all_commits

# Usage example
def main():
    # Initialize collector
    token = os.getenv('GITHUB_TOKEN')  # Store your token in .env file
    if not token:
        token = input("Enter your GitHub token: ")
    
    collector = GitHubDataCollector(token)
    
    # Example: Collect data for a single repository
    owner = "microsoft"
    repo = "vscode"
    
    print(f"Collecting data for {owner}/{repo}...")
    
    # Get repository data
    repo_data = collector.get_repository_data(owner, repo)
    if repo_data:
        print(f"Repository: {repo_data['full_name']}")
        print(f"Stars: {repo_data['stargazers_count']}")
        print(f"Open Issues: {repo_data['open_issues_count']}")
    
    # Get issues data
    issues = collector.get_issues_data(owner, repo)
    print(f"Collected {len(issues)} issues")
    
    # Get contributors data
    contributors = collector.get_contributors_data(owner, repo)
    print(f"Found {len(contributors)} contributors")
    
    # Get recent commits (last 6 months)
    since_date = datetime.now() - timedelta(days=180)
    commits = collector.get_commits_data(owner, repo, since_date)
    print(f"Collected {len(commits)} recent commits")

if __name__ == "__main__":
    main()