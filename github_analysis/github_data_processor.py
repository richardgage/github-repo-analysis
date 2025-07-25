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
    
    def analyze_issue_patterns(self, issues):
        """Analyze issue creation and resolution patterns"""
        if not issues:
            return {}
        
        open_issues = [issue for issue in issues if issue['state'] == 'open']
        closed_issues = [issue for issue in issues if issue['state'] == 'closed']
        
        # Calculate metrics
        total_issues = len(issues)
        open_count = len(open_issues)
        closed_count = len(closed_issues)
        
        # Issue labels analysis
        all_labels = []
        for issue in issues:
            if issue.get('labels'):
                all_labels.extend([label['name'] for label in issue['labels']])
        
        label_counts = Counter(all_labels)
        
        # Time-based analysis
        recent_issues = []
        six_months_ago = datetime.now() - timedelta(days=180)
        
        for issue in issues:
            created_at = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            if created_at > six_months_ago:
                recent_issues.append(issue)
        
        return {
            'total_issues': total_issues,
            'open_issues': open_count,
            'closed_issues': closed_count,
            'open_ratio': open_count / total_issues if total_issues > 0 else 0,
            'recent_issues_count': len(recent_issues),
            'avg_comments_per_issue': np.mean([issue['comments'] for issue in issues]) if issues else 0,
            'most_common_labels': dict(label_counts.most_common(5)),
            'issues_with_assignees': len([i for i in issues if i.get('assignee')]),
            'issues_with_labels': len([i for i in issues if i.get('labels')])
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
    
    def analyze_commit_patterns(self, commits):
        """Analyze commit frequency and patterns"""
        if not commits:
            return {}
        
        # Parse commit dates
        commit_dates = []
        for commit in commits:
            date_str = commit['commit']['author']['date']
            commit_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            commit_dates.append(commit_date)
        
        # Calculate commit frequency
        if commit_dates:
            date_range = max(commit_dates) - min(commit_dates)
            days_active = date_range.days if date_range.days > 0 else 1
            commits_per_day = len(commits) / days_active
            
            # Weekly pattern
            weekly_commits = {}
            for date in commit_dates:
                week = date.strftime('%Y-W%U')
                weekly_commits[week] = weekly_commits.get(week, 0) + 1
            
            avg_weekly_commits = np.mean(list(weekly_commits.values())) if weekly_commits else 0
        else:
            commits_per_day = 0
            avg_weekly_commits = 0
        
        # Analyze commit authors
        authors = [commit['commit']['author']['name'] for commit in commits]
        unique_authors = len(set(authors))
        
        return {
            'total_commits': len(commits),
            'unique_authors': unique_authors,
            'commits_per_day': commits_per_day,
            'avg_weekly_commits': avg_weekly_commits,
            'days_with_activity': len(set(date.date() for date in commit_dates)) if commit_dates else 0
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
    
    for owner, repo in repository_list:
        print(f"Analyzing {owner}/{repo}...")
        
        try:
            # Collect data
            repo_data = collector.get_repository_data(owner, repo)
            issues = collector.get_issues_data(owner, repo)
            contributors = collector.get_contributors_data(owner, repo)
            commits = collector.get_commits_data(owner, repo)
            
            # Analyze
            analysis = analyzer.analyze_repository(repo_data, issues, contributors, commits)
            results.append(analysis)
            
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