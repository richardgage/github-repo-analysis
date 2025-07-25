#!/usr/bin/env python3
"""
github_data_processor.py
GitHub Data Processing and Health Analysis Module

This module contains the RepositoryHealthAnalyzer class for processing
collected GitHub data and calculating repository health metrics.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import time

class RepositoryHealthAnalyzer:
    def __init__(self):
        self.health_metrics = {}
    
    def calculate_response_times(self, issues):
        """Calculate average response time to issues"""
        response_times = []
        
        for issue in issues:
            created_at = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            
            # Get first comment that's not from the issue creator
            if issue['comments'] > 0:
                # You'd need to fetch comments separately for each issue
                # For now, we'll estimate based on patterns
                pass
        
        return response_times
    
    def analyze_issue_patterns(self, issues, time_window_days=180):
        """Analyze issue creation and resolution patterns within time window"""
        if not issues:
            return {}
        
        # Filter issues to time window
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_issues = []
        
        for issue in issues:
            created_at = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            if created_at > cutoff_date:
                recent_issues.append(issue)
        
        # Analyze both all issues and recent issues
        all_open = [issue for issue in issues if issue['state'] == 'open']
        all_closed = [issue for issue in issues if issue['state'] == 'closed']
        
        recent_open = [issue for issue in recent_issues if issue['state'] == 'open']
        recent_closed = [issue for issue in recent_issues if issue['state'] == 'closed']
        
        # Issue labels analysis (from all issues for broader context)
        all_labels = []
        for issue in issues:
            if issue.get('labels'):
                all_labels.extend([label['name'] for label in issue['labels']])
        
        label_counts = Counter(all_labels)
        
        # Calculate resolution rate for recent issues
        recent_total = len(recent_issues)
        recent_resolution_rate = len(recent_closed) / recent_total if recent_total > 0 else 0
        
        return {
            # Overall metrics (for context)
            'total_issues': len(issues),
            'total_open_issues': len(all_open),
            'total_closed_issues': len(all_closed),
            'overall_open_ratio': len(all_open) / len(issues) if issues else 0,
            
            # Recent activity metrics (6-month window)
            'recent_issues_count': recent_total,
            'recent_open_count': len(recent_open),
            'recent_closed_count': len(recent_closed),
            'recent_resolution_rate': recent_resolution_rate,
            'recent_issues_per_day': recent_total / time_window_days,
            
            # Communication metrics
            'avg_comments_per_issue': np.mean([issue['comments'] for issue in issues]) if issues else 0,
            'recent_avg_comments': np.mean([issue['comments'] for issue in recent_issues]) if recent_issues else 0,
            
            # Organization metrics
            'most_common_labels': dict(label_counts.most_common(5)),
            'issues_with_assignees': len([i for i in issues if i.get('assignee')]),
            'issues_with_labels': len([i for i in issues if i.get('labels')]),
            
            # Time window info
            'time_window_days': time_window_days
        }
    
    def analyze_contributor_health(self, contributors, commits):
        """Analyze contributor activity and diversity"""
        if not contributors:
            return {}
        
        total_contributors = len(contributors)
        
        # Analyze contribution distribution
        contributions = [contrib['contributions'] for contrib in contributors]
        
        # Calculate Gini coefficient for contribution inequality
        def gini_coefficient(x):
            sorted_x = sorted(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_x))) / (n * sum(sorted_x))
        
        gini = gini_coefficient(contributions) if len(contributions) > 1 else 0
        
        # Core vs peripheral contributors
        total_contributions = sum(contributions)
        core_contributors = 0
        core_contributions = 0
        
        for contrib in contributions:
            if contrib >= total_contributions * 0.1:  # Contributors with >10% of total contributions
                core_contributors += 1
                core_contributions += contrib
        
        return {
            'total_contributors': total_contributors,
            'core_contributors': core_contributors,
            'core_contribution_ratio': core_contributions / total_contributions if total_contributions > 0 else 0,
            'contribution_gini': gini,
            'avg_contributions': np.mean(contributions) if contributions else 0,
            'median_contributions': np.median(contributions) if contributions else 0,
            'top_contributor_percentage': max(contributions) / total_contributions if total_contributions > 0 else 0
        }
    
    def analyze_commit_patterns(self, commits, time_window_days=180):
        """Analyze commit frequency and patterns within a specific time window"""
        if not commits:
            return {}
        
        # Parse commit dates
        commit_dates = []
        for commit in commits:
            date_str = commit['commit']['author']['date']
            commit_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            commit_dates.append(commit_date)
        
        # Calculate commit frequency over the specified time window
        if commit_dates:
            # Use the full time window for calculation (not just the range of collected commits)
            total_commits = len(commits)
            commits_per_day = total_commits / time_window_days
            
            # Weekly pattern analysis
            weekly_commits = {}
            for date in commit_dates:
                week = date.strftime('%Y-W%U')
                weekly_commits[week] = weekly_commits.get(week, 0) + 1
            
            avg_weekly_commits = np.mean(list(weekly_commits.values())) if weekly_commits else 0
            
            # Active days calculation
            unique_days = len(set(date.date() for date in commit_dates))
            activity_ratio = unique_days / time_window_days  # What percentage of days had commits
        else:
            commits_per_day = 0
            avg_weekly_commits = 0
            activity_ratio = 0
            unique_days = 0
        
        # Analyze commit authors
        authors = [commit['commit']['author']['name'] for commit in commits]
        unique_authors = len(set(authors))
        
        return {
            'total_commits': len(commits),
            'unique_authors': unique_authors,
            'commits_per_day': commits_per_day,
            'avg_weekly_commits': avg_weekly_commits,
            'days_with_activity': unique_days,
            'activity_ratio': activity_ratio,  # New metric: percentage of days with commits
            'time_window_days': time_window_days
        }
    
    def calculate_health_score(self, repo_data, issues_analysis, contributors_analysis, commits_analysis):
        """Calculate overall repository health score (0-100)"""
        score = 0
        max_score = 100
        
        # Activity score (30 points)
        if commits_analysis.get('commits_per_day', 0) > 0.5:
            score += 15
        elif commits_analysis.get('commits_per_day', 0) > 0.1:
            score += 10
        elif commits_analysis.get('commits_per_day', 0) > 0:
            score += 5
        
        if issues_analysis.get('recent_issues_count', 0) > 0:
            score += 15
        
        # Community score (25 points)
        if contributors_analysis.get('total_contributors', 0) > 10:
            score += 15
        elif contributors_analysis.get('total_contributors', 0) > 3:
            score += 10
        elif contributors_analysis.get('total_contributors', 0) > 1:
            score += 5
        
        # Balance score - not too concentrated
        if contributors_analysis.get('contribution_gini', 1) < 0.8:
            score += 10
        
        # Maintenance score (25 points)
        open_ratio = issues_analysis.get('open_ratio', 1)
        if open_ratio < 0.3:
            score += 15
        elif open_ratio < 0.6:
            score += 10
        elif open_ratio < 0.8:
            score += 5
        
        if issues_analysis.get('avg_comments_per_issue', 0) > 1:
            score += 10
        
        # Popularity/Trust score (20 points)
        stars = repo_data.get('stargazers_count', 0) if repo_data else 0
        if stars > 1000:
            score += 20
        elif stars > 100:
            score += 15
        elif stars > 10:
            score += 10
        elif stars > 0:
            score += 5
        
        return min(score, max_score)
    
    def analyze_repository(self, repo_data, issues, contributors, commits):
        """Complete repository analysis"""
        issues_analysis = self.analyze_issue_patterns(issues)
        contributors_analysis = self.analyze_contributor_health(contributors, commits)
        commits_analysis = self.analyze_commit_patterns(commits)
        
        health_score = self.calculate_health_score(
            repo_data, issues_analysis, contributors_analysis, commits_analysis
        )
        
        return {
            'repository': repo_data.get('full_name', 'Unknown') if repo_data else 'Unknown',
            'health_score': health_score,
            'basic_stats': {
                'stars': repo_data.get('stargazers_count', 0) if repo_data else 0,
                'forks': repo_data.get('forks_count', 0) if repo_data else 0,
                'open_issues': repo_data.get('open_issues_count', 0) if repo_data else 0,
                'language': repo_data.get('language', 'Unknown') if repo_data else 'Unknown',
                'created_at': repo_data.get('created_at', '') if repo_data else '',
                'updated_at': repo_data.get('updated_at', '') if repo_data else ''
            },
            'issues_metrics': issues_analysis,
            'contributors_metrics': contributors_analysis,
            'commits_metrics': commits_analysis
        }

# Example usage function
def analyze_multiple_repositories(collector, repository_list):
    """Analyze multiple repositories and create a dataset"""
    analyzer = RepositoryHealthAnalyzer()
    results = []
    
    repo_count = 0

    for owner, repo in repository_list:
        print(f"#{repo_count}: Analyzing {owner}/{repo}...")
        
        try:
            # Collect data
            repo_data = collector.get_repository_data(owner, repo)
            issues = collector.get_issues_data(owner, repo)
            contributors = collector.get_contributors_data(owner, repo)
            commits = collector.get_commits_data(owner, repo)
            
            # Analyze
            analysis = analyzer.analyze_repository(repo_data, issues, contributors, commits)
            results.append(analysis)
            repo_count += 1
            
            # Be nice to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"Error analyzing {owner}/{repo}: {str(e)}")
            continue
    
    return results

# Sample repository list for testing
sample_repos = [
    ('microsoft', 'vscode'),
    ('facebook', 'react'),
    ('tensorflow', 'tensorflow'),
    ('nodejs', 'node'),
    ('python', 'cpython')
]