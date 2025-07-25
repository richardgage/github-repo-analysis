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
    Get a sample of repositories to analyze
    You can modify this list based on your research needs
    """
    # Popular repositories from different languages and domains
    sample_repos = [
        # JavaScript/Web Development
        ('facebook', 'react'),
        ('vuejs', 'vue'),
        ('angular', 'angular'),
        ('nodejs', 'node'),
        ('expressjs', 'express'),
        
        # Python
        ('python', 'cpython'),
        ('pallets', 'flask'),
        ('django', 'django'),
        ('numpy', 'numpy'),
        ('pandas-dev', 'pandas'),
        
        # Java
        ('spring-projects', 'spring-boot'),
        ('elastic', 'elasticsearch'),
        ('apache', 'kafka'),
        
        # C/C++
        ('torvalds', 'linux'),
        ('microsoft', 'terminal'),
        
        # Go
        ('golang', 'go'),
        ('kubernetes', 'kubernetes'),
        ('docker', 'docker'),
        
        # Machine Learning
        ('tensorflow', 'tensorflow'),
        ('pytorch', 'pytorch'),
        ('scikit-learn', 'scikit-learn'),
        
        # Developer Tools
        ('microsoft', 'vscode'),
        ('git', 'git'),
        ('github', 'cli'),
        
        # Mobile
        ('flutter', 'flutter'),
        ('facebook', 'react-native'),
        
        # Data Science
        ('jupyter', 'notebook'),
        ('plotly', 'plotly.js'),
        
        # Smaller/Medium Projects for comparison
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
                'open_ratio': result['issues_metrics'].get('open_ratio', 0),
                'avg_comments_per_issue': result['issues_metrics'].get('avg_comments_per_issue', 0),
                'commits_per_day': result['commits_metrics'].get('commits_per_day', 0),
                'core_contributors': result['contributors_metrics'].get('core_contributors', 0),
                'contribution_gini': result['contributors_metrics'].get('contribution_gini', 0),
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
    estimated_calls = len(repositories) * 15  # Rough estimate
    
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