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
        if not contributors or len(contributors) == 0:
            print("Warning: No contributors data available")
            return {
                'total_contributors': 0,
                'core_contributors': 0,
                'core_contribution_ratio': 0,
                'contribution_gini': 0,
                'avg_contributions': 0,
                'median_contributions': 0,
                'top_contributor_percentage': 0
            }
        
        try:
            total_contributors = len(contributors)
            
            # Analyze contribution distribution
            contributions = []
            for contrib in contributors:
                if isinstance(contrib, dict) and 'contributions' in contrib:
                    contributions.append(contrib['contributions'])
                else:
                    print(f"Warning: Invalid contributor data: {contrib}")
            
            if not contributions:
                print("Warning: No valid contribution data found")
                return {
                    'total_contributors': total_contributors,
                    'core_contributors': 0,
                    'core_contribution_ratio': 0,
                    'contribution_gini': 0,
                    'avg_contributions': 0,
                    'median_contributions': 0,
                    'top_contributor_percentage': 0
                }
            
            # Calculate Gini coefficient for contribution inequality
            def gini_coefficient(x):
                if len(x) <= 1:
                    return 0
                sorted_x = sorted(x)
                n = len(x)
                cumsum = np.cumsum(sorted_x)
                return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_x))) / (n * sum(sorted_x))
            
            gini = gini_coefficient(contributions) if len(contributions) > 1 else 0
            
            # Core vs peripheral contributors
            total_contributions = sum(contributions)
            core_contributors = 0
            core_contributions = 0
            
            if total_contributions > 0:
                for contrib in contributions:
                    if contrib >= total_contributions * 0.1:  # Contributors with >10% of total contributions
                        core_contributors += 1
                        core_contributions += contrib
                
                core_ratio = core_contributions / total_contributions
                top_percentage = max(contributions) / total_contributions
            else:
                core_ratio = 0
                top_percentage = 0
            
            return {
                'total_contributors': total_contributors,
                'core_contributors': core_contributors,
                'core_contribution_ratio': core_ratio,
                'contribution_gini': gini,
                'avg_contributions': np.mean(contributions) if contributions else 0,
                'median_contributions': np.median(contributions) if contributions else 0,
                'top_contributor_percentage': top_percentage
            }
            
        except Exception as e:
            print(f"Error in contributor analysis: {e}")
            return {
                'total_contributors': len(contributors) if contributors else 0,
                'core_contributors': 0,
                'core_contribution_ratio': 0,
                'contribution_gini': 0,
                'avg_contributions': 0,
                'median_contributions': 0,
                'top_contributor_percentage': 0
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
        
    def analyze_issue_response_times(self, issues, collector, owner, repo):
        """Analyze how quickly issues get first responses"""
        if not issues or len(issues) < 5:  # Skip if too few issues
            return {
                'avg_response_time_hours': 0,
                'median_response_time_hours': 0,
                'response_rate': 0,
                'sample_size': 0
            }
        
        response_times = []
        responded_count = 0
        
        # Sample recent issues to avoid too many API calls
        sample_issues = issues[:10]  # Check first 10 issues for response times
        
        for issue in sample_issues:
            if issue['comments'] > 0:
                try:
                    comments = collector.get_issue_comments(owner, repo, issue['number'])
                    if comments:
                        issue_created = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                        first_comment = datetime.strptime(comments[0]['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                        
                        # Only count if comment is not from issue creator
                        if comments[0]['user']['login'] != issue['user']['login']:
                            response_time = (first_comment - issue_created).total_seconds() / 3600  # hours
                            response_times.append(response_time)
                            responded_count += 1
                    
                    time.sleep(0.2)  # Be gentle with API
                except Exception as e:
                    continue
        
        if response_times:
            avg_response = np.mean(response_times)
            median_response = np.median(response_times)
        else:
            avg_response = 0
            median_response = 0
        
        response_rate = responded_count / len(sample_issues) if sample_issues else 0
        
        return {
            'avg_response_time_hours': avg_response,
            'median_response_time_hours': median_response,
            'response_rate': response_rate,
            'sample_size': len(sample_issues)
        }
    
    def analyze_release_patterns(self, releases):
        """Analyze release frequency and patterns"""
        if not releases:
            return {
                'total_releases': 0,
                'releases_per_month': 0,
                'days_since_last_release': 0,
                'avg_days_between_releases': 0,
                'release_consistency': 0
            }
        
        # Parse release dates
        release_dates = []
        for release in releases:
            if not release.get('draft', False):  # Skip draft releases
                date_str = release.get('published_at') or release.get('created_at')
                if date_str:
                    release_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
                    release_dates.append(release_date)
        
        if not release_dates:
            return {
                'total_releases': 0,
                'releases_per_month': 0,
                'days_since_last_release': 0,
                'avg_days_between_releases': 0,
                'release_consistency': 0
            }
        
        release_dates.sort(reverse=True)  # Most recent first
        
        # Calculate metrics
        total_releases = len(release_dates)
        days_since_last = (datetime.now() - release_dates[0]).days
        
        # Calculate average time between releases
        if len(release_dates) > 1:
            time_spans = []
            for i in range(len(release_dates) - 1):
                span = (release_dates[i] - release_dates[i + 1]).days
                time_spans.append(span)
            
            avg_days_between = np.mean(time_spans)
            # Consistency: lower standard deviation = more consistent
            consistency = 1 / (1 + np.std(time_spans) / avg_days_between) if avg_days_between > 0 else 0
        else:
            avg_days_between = 0
            consistency = 0
        
        # Calculate releases per month based on project age
        if len(release_dates) > 1:
            project_age_days = (release_dates[0] - release_dates[-1]).days
            releases_per_month = (total_releases / project_age_days) * 30 if project_age_days > 0 else 0
        else:
            releases_per_month = 0
        
        return {
            'total_releases': total_releases,
            'releases_per_month': releases_per_month,
            'days_since_last_release': days_since_last,
            'avg_days_between_releases': avg_days_between,
            'release_consistency': consistency
        }
    
    def analyze_documentation_quality(self, repo_data, contents):
        """Assess documentation quality based on key files"""
        try:
            if not contents or not isinstance(contents, list):
                print("Warning: No repository contents available for documentation analysis")
                return {
                    'documentation_score': 20,  # Give some points for having a repo
                    'has_readme': False,
                    'has_contributing': False,
                    'has_license': False,
                    'has_changelog': False,
                    'readme_length': 0
                }
            
            # Look for key documentation files
            file_names = []
            for item in contents:
                if isinstance(item, dict) and item.get('type') == 'file':
                    file_names.append(item.get('name', '').lower())
            
            has_readme = any('readme' in name for name in file_names)
            has_contributing = any('contributing' in name for name in file_names)
            has_license = any('license' in name or 'licence' in name for name in file_names)
            has_changelog = any('changelog' in name or 'history' in name for name in file_names)
            
            # Get README length if available
            readme_length = 0
            if repo_data and 'description' in repo_data and repo_data['description']:
                readme_length = len(repo_data['description'])
            
            # Calculate documentation score
            doc_score = 0
            if has_readme: doc_score += 40
            if has_contributing: doc_score += 25
            if has_license: doc_score += 20
            if has_changelog: doc_score += 15
            
            return {
                'documentation_score': doc_score,
                'has_readme': has_readme,
                'has_contributing': has_contributing,
                'has_license': has_license,
                'has_changelog': has_changelog,
                'readme_length': readme_length
            }
            
        except Exception as e:
            print(f"Error in documentation analysis: {e}")
            return {
                'documentation_score': 20,  # Default score
                'has_readme': False,
                'has_contributing': False,
                'has_license': False,
                'has_changelog': False,
                'readme_length': 0
            }
    
    def analyze_pull_request_patterns(self, pull_requests):
        """Analyze pull request acceptance and review patterns"""
        if not pull_requests:
            return {
                'total_prs': 0,
                'pr_acceptance_rate': 0,
                'avg_pr_comments': 0,
                'external_pr_rate': 0,
                'avg_time_to_merge_days': 0
            }
        
        total_prs = len(pull_requests)
        merged_prs = 0
        external_prs = 0
        total_comments = 0
        merge_times = []
        
        for pr in pull_requests:
            # Count merged PRs
            if pr.get('merged_at'):
                merged_prs += 1
                
                # Calculate time to merge
                created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                merged_at = datetime.strptime(pr['merged_at'], '%Y-%m-%dT%H:%M:%SZ')
                merge_time = (merged_at - created_at).days
                merge_times.append(merge_time)
            
            # Count external PRs (from non-members)
            # Note: This is a simplified check - in reality you'd need to check organization membership
            total_comments += pr.get('comments', 0) + pr.get('review_comments', 0)
        
        acceptance_rate = merged_prs / total_prs if total_prs > 0 else 0
        avg_comments = total_comments / total_prs if total_prs > 0 else 0
        avg_merge_time = np.mean(merge_times) if merge_times else 0
        
        # External PR rate is harder to calculate without organization info
        # For now, we'll estimate based on PR patterns
        external_pr_rate = 0.5  # Placeholder - would need additional API calls to determine accurately
        
        return {
            'total_prs': total_prs,
            'pr_acceptance_rate': acceptance_rate,
            'avg_pr_comments': avg_comments,
            'external_pr_rate': external_pr_rate,
            'avg_time_to_merge_days': avg_merge_time
        }
    
    def analyze_repository(self, repo_data, issues, contributors, commits, releases=None, pull_requests=None, contents=None, collector=None, owner=None, repo=None):
        """Complete repository analysis with new metrics and error handling"""
        try:
            issues_analysis = self.analyze_issue_patterns(issues)
        except Exception as e:
            print(f"Error analyzing issues: {e}")
            issues_analysis = {}
        
        try:
            contributors_analysis = self.analyze_contributor_health(contributors, commits)
        except Exception as e:
            print(f"Error analyzing contributors: {e}")
            contributors_analysis = {}
        
        try:
            commits_analysis = self.analyze_commit_patterns(commits)
        except Exception as e:
            print(f"Error analyzing commits: {e}")
            commits_analysis = {}
        
        # New high-priority metrics with error handling
        response_analysis = {}
        if collector and owner and repo and issues:
            try:
                response_analysis = self.analyze_issue_response_times(issues, collector, owner, repo)
            except Exception as e:
                print(f"Error analyzing response times: {e}")
                response_analysis = {}
        
        try:
            release_analysis = self.analyze_release_patterns(releases) if releases else {}
        except Exception as e:
            print(f"Error analyzing releases: {e}")
            release_analysis = {}
        
        try:
            documentation_analysis = self.analyze_documentation_quality(repo_data, contents)
        except Exception as e:
            print(f"Error analyzing documentation: {e}")
            documentation_analysis = {}
        
        try:
            pr_analysis = self.analyze_pull_request_patterns(pull_requests) if pull_requests else {}
        except Exception as e:
            print(f"Error analyzing pull requests: {e}")
            pr_analysis = {}
        
        try:
            health_score = self.calculate_health_score(
                repo_data, issues_analysis, contributors_analysis, commits_analysis
            )
        except Exception as e:
            print(f"Error calculating health score: {e}")
            health_score = 0
        
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
            'commits_metrics': commits_analysis,
            'response_metrics': response_analysis,
            'release_metrics': release_analysis,
            'documentation_metrics': documentation_analysis,
            'pr_metrics': pr_analysis
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