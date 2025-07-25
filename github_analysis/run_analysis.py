#!/usr/bin/env python3
"""
run_analysis.py
Main script to run GitHub repository health analysis

This script uses the GitHubDataCollector and RepositoryHealthAnalyzer
to analyze multiple repositories and save results for research.
"""

import pandas as pd
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Import our custom classes (make sure the files are in the same folder)
from github_collector import GitHubDataCollector
from github_data_processor import RepositoryHealthAnalyzer, analyze_multiple_repositories

def get_sample_repositories():
    """
    Get a sample of 50 repositories to analyze
    Expanded to cover more languages, domains, and project sizes
    """
    sample_repos = [
        # JavaScript/Web Development (8 repos)
        ('facebook', 'react'),
        ('vuejs', 'vue'),
        ('angular', 'angular'),
        ('nodejs', 'node'),
        ('expressjs', 'express'),
        ('webpack', 'webpack'),
        ('facebook', 'create-react-app'),
        ('vercel', 'next.js'),
        
        # Python (8 repos)
        ('python', 'cpython'),
        ('pallets', 'flask'),
        ('django', 'django'),
        ('numpy', 'numpy'),
        ('pandas-dev', 'pandas'),
        ('psf', 'requests'),
        ('pytest-dev', 'pytest'),
        ('scikit-learn', 'scikit-learn'),
        
        # Java (5 repos)
        ('spring-projects', 'spring-boot'),
        ('elastic', 'elasticsearch'),
        ('apache', 'kafka'),
        ('google', 'guava'),
        ('ReactiveX', 'RxJava'),
        
        # C/C++ (4 repos)
        ('torvalds', 'linux'),
        ('microsoft', 'terminal'),
        ('redis', 'redis'),
        ('nginx', 'nginx'),
        
        # Go (5 repos)
        ('golang', 'go'),
        ('kubernetes', 'kubernetes'),
        ('docker', 'docker'),
        ('prometheus', 'prometheus'),
        ('grafana', 'grafana'),
        
        # Machine Learning/AI (5 repos)
        ('tensorflow', 'tensorflow'),
        ('pytorch', 'pytorch'),
        ('huggingface', 'transformers'),
        ('microsoft', 'ML-For-Beginners'),
        ('openai', 'whisper'),
        
        # Developer Tools (5 repos)
        ('microsoft', 'vscode'),
        ('git', 'git'),
        ('github', 'cli'),
        ('atom', 'atom'),
        ('neovim', 'neovim'),
        
        # Mobile Development (3 repos)
        ('flutter', 'flutter'),
        ('facebook', 'react-native'),
        ('ionic-team', 'ionic-framework'),
        
        # Data Science/Analytics (3 repos)
        ('jupyter', 'notebook'),
        ('plotly', 'plotly.js'),
        ('apache', 'superset'),
        
        # Rust (2 repos)
        ('rust-lang', 'rust'),
        ('denoland', 'deno'),
        
        # Small/Medium Projects for comparison (2 repos)
        ('octocat', 'Hello-World'),
        ('github', 'docs'),
    ]
    
    return sample_repos

def save_results(results, base_filename="github_analysis"):
    """Save results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON
    json_filename = f"{base_filename}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {json_filename}")
    
    # Create simplified CSV for easy analysis
    csv_data = []
    for result in results:
        if result:  # Skip any failed analyses
            csv_row = {
                'repository': result['repository'],
                'health_score': result['health_score'],
                'stars': result['basic_stats']['stars'],
                'forks': result['basic_stats']['forks'],
                'open_issues': result['basic_stats']['open_issues'],
                'language': result['basic_stats']['language'],
                'total_contributors': result['contributors_metrics'].get('total_contributors', 0),
                'total_commits': result['commits_metrics'].get('total_commits', 0),
                'open_ratio': result['issues_metrics'].get('overall_open_ratio', 0),
                'recent_issues_count': result['issues_metrics'].get('recent_issues_count', 0),
                'recent_issues_per_day': result['issues_metrics'].get('recent_issues_per_day', 0),
                'recent_resolution_rate': result['issues_metrics'].get('recent_resolution_rate', 0),
                'avg_comments_per_issue': result['issues_metrics'].get('avg_comments_per_issue', 0),
                'commits_per_day': result['commits_metrics'].get('commits_per_day', 0),
                'activity_ratio': result['commits_metrics'].get('activity_ratio', 0),
                'time_window_days': result['commits_metrics'].get('time_window_days', 180),
                'core_contributors': result['contributors_metrics'].get('core_contributors', 0),
                'contribution_gini': result['contributors_metrics'].get('contribution_gini', 0),
                
                # New high-priority metrics
                'avg_response_time_hours': result['response_metrics'].get('avg_response_time_hours', 0),
                'response_rate': result['response_metrics'].get('response_rate', 0),
                'total_releases': result['release_metrics'].get('total_releases', 0),
                'releases_per_month': result['release_metrics'].get('releases_per_month', 0),
                'days_since_last_release': result['release_metrics'].get('days_since_last_release', 0),
                'documentation_score': result['documentation_metrics'].get('documentation_score', 0),
                'has_readme': result['documentation_metrics'].get('has_readme', False),
                'has_contributing': result['documentation_metrics'].get('has_contributing', False),
                'has_license': result['documentation_metrics'].get('has_license', False),
                'pr_acceptance_rate': result['pr_metrics'].get('pr_acceptance_rate', 0),
                'avg_time_to_merge_days': result['pr_metrics'].get('avg_time_to_merge_days', 0)
            }
            csv_data.append(csv_row)
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    csv_filename = f"{base_filename}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✓ CSV summary saved to: {csv_filename}")
    
    # Print quick summary
    if csv_data:
        print(f"\n📊 Analysis Summary:")
        print(f"   Repositories analyzed: {len(csv_data)}")
        print(f"   Average health score: {df['health_score'].mean():.1f}")
        print(f"   Highest scoring repo: {df.loc[df['health_score'].idxmax(), 'repository']}")
        print(f"   Most stars: {df.loc[df['stars'].idxmax(), 'repository']} ({df['stars'].max():,} stars)")
    
    return json_filename, csv_filename

def main():
    """Main analysis function"""
    print("🚀 Starting GitHub Repository Health Analysis")
    print("=" * 60)
    
    # Load environment and set up collector
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    
    if not token:
        print("❌ ERROR: No GitHub token found in .env file")
        print("Please create a .env file with: GITHUB_TOKEN=your_token_here")
        return
    
    print(f"✓ GitHub token loaded")
    
    # Initialize collector
    collector = GitHubDataCollector(token)
    
    # Test connection
    print("🔍 Testing API connection...")
    test_response = collector.make_request(f"{collector.base_url}/rate_limit")
    if not test_response:
        print("❌ Failed to connect to GitHub API")
        return
    
    remaining_calls = test_response['rate']['remaining']
    print(f"✓ API connected. {remaining_calls} calls remaining")
    
    # Get repositories to analyze
    repositories = get_sample_repositories()
    estimated_calls = len(repositories) * 25  # Updated estimate for more API calls
    
    print(f"\n📋 Analysis Plan:")
    print(f"   Repositories to analyze: {len(repositories)}")
    print(f"   Estimated API calls needed: {estimated_calls}")
    print(f"   Available API calls: {remaining_calls}")
    
    if estimated_calls > remaining_calls:
        print("⚠️  WARNING: This might exceed your API rate limit")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Analysis cancelled")
            return
    
    # Run the analysis
    print(f"\n🔬 Starting analysis...")
    start_time = time.time()
    
    results = analyze_multiple_repositories(collector, repositories)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ Analysis completed in {duration:.1f} seconds")
    
    # Save results
    if results:
        json_file, csv_file = save_results(results)
        
        print(f"\n📁 Files created:")
        print(f"   📄 {json_file} (detailed data)")
        print(f"   📊 {csv_file} (summary for analysis)")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Open {csv_file} in Excel or Python for analysis")
        print(f"   2. Look for correlations between variables")
        print(f"   3. Test your hypotheses about contributors vs issues")
        print(f"   4. Create visualizations for your report")
        
    else:
        print("❌ No results collected. Check for errors above.")

if __name__ == "__main__":
    main()