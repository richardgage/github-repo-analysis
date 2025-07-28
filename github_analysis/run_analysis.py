#!/usr/bin/env python3
"""
run_analysis_200.py - 200 Repository Analysis for Regression Models
Main script to analyze 200 GitHub repositories for statistical modeling

This provides sufficient statistical power for:
- Multiple regression analysis
- Machine learning models
- Subgroup analysis by language/domain
- Robust hypothesis testing
"""

import pandas as pd
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Import our custom classes
from github_collector import GitHubDataCollector
from github_data_processor import RepositoryHealthAnalyzer, analyze_multiple_repositories

def get_sample_repositories():
    """
    Get exactly 200 repositories for robust regression analysis
    Comprehensive coverage across languages, domains, and project sizes
    """
    sample_repos = [
        # JavaScript/Web Development (25 repos)
        ('facebook', 'react'),
        ('vuejs', 'vue'),
        ('angular', 'angular'),
        ('nodejs', 'node'),
        ('expressjs', 'express'),
        ('webpack', 'webpack'),
        ('facebook', 'create-react-app'),
        ('vercel', 'next.js'),
        ('sveltejs', 'svelte'),
        ('facebook', 'jest'),
        ('typicode', 'json-server'),
        ('lodash', 'lodash'),
        ('moment', 'moment'),
        ('chartjs', 'Chart.js'),
        ('axios', 'axios'),
        ('jquery', 'jquery'),
        ('d3', 'd3'),
        ('twbs', 'bootstrap'),
        ('facebook', 'metro'),
        ('gulpjs', 'gulp'),
        ('gruntjs', 'grunt'),
        ('yarnpkg', 'yarn'),
        ('parcel-bundler', 'parcel'),
        ('rollup', 'rollup'),
        ('storybookjs', 'storybook'),
        
        # Python (25 repos)
        ('python', 'cpython'),
        ('pallets', 'flask'),
        ('django', 'django'),
        ('numpy', 'numpy'),
        ('pandas-dev', 'pandas'),
        ('psf', 'requests'),
        ('pytest-dev', 'pytest'),
        ('scikit-learn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('encode', 'django-rest-framework'),
        ('celery', 'celery'),
        ('sqlalchemy', 'sqlalchemy'),
        ('joke2k', 'faker'),
        ('python-pillow', 'Pillow'),
        ('tornadoweb', 'tornado'),
        ('bottlepy', 'bottle'),
        ('pypa', 'pip'),
        ('pypa', 'setuptools'),
        ('pypa', 'virtualenv'),
        ('boto', 'boto3'),
        ('paramiko', 'paramiko'),
        ('fabric', 'fabric'),
        ('ansible', 'ansible'),
        ('scrapy', 'scrapy'),
        
        # Java (20 repos)
        ('spring-projects', 'spring-boot'),
        ('elastic', 'elasticsearch'),
        ('apache', 'kafka'),
        ('google', 'guava'),
        ('ReactiveX', 'RxJava'),
        ('junit-team', 'junit5'),
        ('mockito', 'mockito'),
        ('square', 'okhttp'),
        ('apache', 'maven'),
        ('gradle', 'gradle'),
        ('apache', 'tomcat'),
        ('apache', 'commons-lang'),
        ('jackson-core', 'jackson-core'),
        ('FasterXML', 'jackson-databind'),
        ('square', 'retrofit'),
        ('google', 'gson'),
        ('apache', 'lucene-solr'),
        ('hibernate', 'hibernate-orm'),
        ('mybatis', 'mybatis-3'),
        ('spring-projects', 'spring-framework'),
        
        # C/C++ (18 repos)
        ('torvalds', 'linux'),
        ('microsoft', 'terminal'),
        ('redis', 'redis'),
        ('nginx', 'nginx'),
        ('curl', 'curl'),
        ('git', 'git'),
        ('libuv', 'libuv'),
        ('sqlite', 'sqlite'),
        ('opencv', 'opencv'),
        ('microsoft', 'vcpkg'),
        ('jemalloc', 'jemalloc'),
        ('memcached', 'memcached'),
        ('gperftools', 'gperftools'),
        ('google', 'leveldb'),
        ('facebook', 'rocksdb'),
        ('zeromq', 'libzmq'),
        ('libevent', 'libevent'),
        ('FFmpeg', 'FFmpeg'),
        
        # Go (18 repos)
        ('golang', 'go'),
        ('kubernetes', 'kubernetes'),
        ('docker', 'docker'),
        ('prometheus', 'prometheus'),
        ('grafana', 'grafana'),
        ('moby', 'moby'),
        ('helm', 'helm'),
        ('hashicorp', 'terraform'),
        ('etcd-io', 'etcd'),
        ('influxdata', 'influxdb'),
        ('containerd', 'containerd'),
        ('istio', 'istio'),
        ('cockroachdb', 'cockroach'),
        ('hashicorp', 'consul'),
        ('hashicorp', 'vault'),
        ('coreos', 'flannel'),
        ('gorilla', 'mux'),
        ('gin-gonic', 'gin'),
        
        # Machine Learning/AI (15 repos)
        ('tensorflow', 'tensorflow'),
        ('pytorch', 'pytorch'),
        ('huggingface', 'transformers'),
        ('microsoft', 'ML-For-Beginners'),
        ('openai', 'whisper'),
        ('keras-team', 'keras'),
        ('apache', 'spark'),
        ('streamlit', 'streamlit'),
        ('mlflow', 'mlflow'),
        ('microsoft', 'nni'),
        ('ray-project', 'ray'),
        ('dmlc', 'xgboost'),
        ('catboost', 'catboost'),
        ('microsoft', 'LightGBM'),
        ('optuna', 'optuna'),
        
        # Developer Tools (15 repos)
        ('microsoft', 'vscode'),
        ('github', 'cli'),
        ('atom', 'atom'),
        ('neovim', 'neovim'),
        ('prettier', 'prettier'),
        ('eslint', 'eslint'),
        ('babel', 'babel'),
        ('postcss', 'postcss'),
        ('stylelint', 'stylelint'),
        ('jshint', 'jshint'),
        ('facebook', 'watchman'),
        ('nodemon', 'nodemon'),
        ('browserify', 'browserify'),
        ('shelljs', 'shelljs'),
        ('commitizen', 'cz-cli'),
        
        # Mobile Development (12 repos)
        ('flutter', 'flutter'),
        ('facebook', 'react-native'),
        ('ionic-team', 'ionic-framework'),
        ('apache', 'cordova'),
        ('xamarin', 'xamarin-forms'),
        ('nativescript', 'NativeScript'),
        ('expo', 'expo'),
        ('phonegap', 'phonegap'),
        ('quasar', 'quasar'),
        ('onsen', 'OnsenUI'),
        ('framework7io', 'framework7'),
        ('mobile-shell', 'mosh'),
        
        # Data Science/Analytics (10 repos)
        ('jupyter', 'notebook'),
        ('plotly', 'plotly.js'),
        ('apache', 'superset'),
        ('apache', 'airflow'),
        ('dask', 'dask'),
        ('bokeh', 'bokeh'),
        ('altair-viz', 'altair'),
        ('pydata', 'xarray'),
        ('vaexio', 'vaex'),
        ('rapidsai', 'cudf'),
        
        # Rust (10 repos)
        ('rust-lang', 'rust'),
        ('denoland', 'deno'),
        ('servo', 'servo'),
        ('tokio-rs', 'tokio'),
        ('serde-rs', 'serde'),
        ('clap-rs', 'clap'),
        ('diesel-rs', 'diesel'),
        ('actix', 'actix-web'),
        ('hyperium', 'hyper'),
        ('rustwasm', 'wasm-pack'),
        
        # PHP (8 repos)
        ('laravel', 'laravel'),
        ('symfony', 'symfony'),
        ('composer', 'composer'),
        ('phpunit', 'phpunit'),
        ('doctrine', 'orm'),
        ('guzzle', 'guzzle'),
        ('sebastianbergmann', 'phpunit'),
        ('cakephp', 'cakephp'),
        
        # Ruby (8 repos)
        ('rails', 'rails'),
        ('jekyll', 'jekyll'),
        ('rspec', 'rspec-core'),
        ('rubygems', 'rubygems'),
        ('bundler', 'bundler'),
        ('sinatra', 'sinatra'),
        ('puma', 'puma'),
        ('sidekiq', 'sidekiq'),
        
        # Swift/iOS (7 repos)
        ('apple', 'swift'),
        ('Alamofire', 'Alamofire'),
        ('realm', 'realm-swift'),
        ('SwiftyJSON', 'SwiftyJSON'),
        ('SnapKit', 'SnapKit'),
        ('RxSwiftCommunity', 'RxSwift'),
        ('vapor', 'vapor'),
        
        # Databases (5 repos)
        ('mongodb', 'mongo'),
        ('postgres', 'postgres'),
        ('mysql', 'mysql-server'),
        ('MariaDB', 'server'),
        ('apache', 'cassandra'),
        
        # DevOps/Infrastructure (4 repos)
        ('saltstack', 'salt'),
        ('chef', 'chef'),
        ('puppet', 'puppet'),
        ('hashicorp', 'packer'),
    ]
    
    # Verify we have exactly 200 repositories
    print(f"Total repositories in list: {len(sample_repos)}")
    assert len(sample_repos) == 200, f"Expected 200 repos, got {len(sample_repos)}"
    
    return sample_repos

def save_results(results, base_filename="github_analysis_200"):
    """Save results optimized for regression analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON
    json_filename = f"{base_filename}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {json_filename}")
    
    # Create regression-ready CSV
    csv_data = []
    for result in results:
        if result:  # Skip any failed analyses
            csv_row = {
                # Dependent variable candidates
                'health_score': result['health_score'],
                'open_issues': result['basic_stats']['open_issues'],
                'open_ratio': result['issues_metrics'].get('overall_open_ratio', 0),
                
                # Independent variables - Repository characteristics
                'repository': result['repository'],
                'stars': result['basic_stats']['stars'],
                'forks': result['basic_stats']['forks'],
                'language': result['basic_stats']['language'],
                
                # Contributor variables
                'total_contributors': result['contributors_metrics'].get('total_contributors', 0),
                'core_contributors': result['contributors_metrics'].get('core_contributors', 0),
                'contribution_gini': result['contributors_metrics'].get('contribution_gini', 0),
                'top_contributor_percentage': result['contributors_metrics'].get('top_contributor_percentage', 0),
                
                # Activity variables
                'total_commits': result['commits_metrics'].get('total_commits', 0),
                'commits_per_day': result['commits_metrics'].get('commits_per_day', 0),
                'activity_ratio': result['commits_metrics'].get('activity_ratio', 0),
                'unique_authors': result['commits_metrics'].get('unique_authors', 0),
                
                # Issue management variables
                'recent_issues_count': result['issues_metrics'].get('recent_issues_count', 0),
                'recent_issues_per_day': result['issues_metrics'].get('recent_issues_per_day', 0),
                'recent_resolution_rate': result['issues_metrics'].get('recent_resolution_rate', 0),
                'avg_comments_per_issue': result['issues_metrics'].get('avg_comments_per_issue', 0),
                
                # Derived variables for regression
                'log_stars': np.log1p(result['basic_stats']['stars']),  # Log transform for skewed data
                'log_contributors': np.log1p(result['contributors_metrics'].get('total_contributors', 0)),
                'contributors_per_star': (result['contributors_metrics'].get('total_contributors', 0) / 
                                        max(result['basic_stats']['stars'], 1)),
                'commits_per_contributor': (result['commits_metrics'].get('total_commits', 0) / 
                                          max(result['contributors_metrics'].get('total_contributors', 0), 1)),
                
                # Time window
                'time_window_days': result['commits_metrics'].get('time_window_days', 180),
            }
            csv_data.append(csv_row)
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    csv_filename = f"{base_filename}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✓ Regression-ready CSV saved to: {csv_filename}")
    
    # Print regression analysis summary
    if csv_data:
        valid_scores = [row['health_score'] for row in csv_data if row['health_score'] is not None and not pd.isna(row['health_score'])]
        
        print(f"\n📊 200-Repository Regression Dataset Summary:")
        print(f"   Total repositories: {len(csv_data)}")
        print(f"   Valid health scores: {len(valid_scores)}")
        
        if valid_scores:
            print(f"   Health score range: {min(valid_scores):.1f} - {max(valid_scores):.1f}")
            print(f"   Mean health score: {np.mean(valid_scores):.1f}")
            print(f"   Std health score: {np.std(valid_scores):.1f}")
            
            # Language distribution for regression
            languages = [row['language'] for row in csv_data if row['language']]
            lang_counts = pd.Series(languages).value_counts()
            print(f"\n   Language distribution (top 10):")
            for lang, count in lang_counts.head(10).items():
                print(f"     {lang}: {count} repos")
            
            # Variable correlations preview
            numeric_cols = ['health_score', 'stars', 'total_contributors', 'commits_per_day', 'open_ratio']
            temp_df = pd.DataFrame(csv_data)[numeric_cols]
            correlations = temp_df.corr()['health_score'].drop('health_score')
            print(f"\n   Key correlations with health_score:")
            for var, corr in correlations.items():
                print(f"     {var}: {corr:.3f}")
    
    return json_filename, csv_filename

def main():
    """Main analysis function for 200 repositories"""
    print("🚀 Starting 200-Repository Analysis for Regression Models")
    print("=" * 70)
    
    # Load environment and set up collector
    load_dotenv()
    token = os.getenv('GITHUB_TOKEN')
    
    if not token:
        print("❌ ERROR: No GitHub token found in .env file")
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
    estimated_calls = len(repositories) * 15  # ~3000 calls for 200 repos
    
    print(f"\n📋 200-Repository Regression Analysis Plan:")
    print(f"   Repositories to analyze: {len(repositories)}")
    print(f"   Estimated API calls needed: {estimated_calls}")
    print(f"   Available API calls: {remaining_calls}")
    print(f"   Estimated runtime: {len(repositories) * 0.75:.0f} minutes")
    
    if estimated_calls > remaining_calls:
        print("⚠️  WARNING: This might exceed your API rate limit")
        print("   Consider running in batches or waiting for rate limit reset")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Analysis cancelled")
            return
    
    # Run the analysis
    print(f"\n🔬 Starting 200-repository analysis for regression modeling...")
    print(f"   This will provide excellent statistical power for:")
    print(f"   • Multiple regression analysis")
    print(f"   • Machine learning models") 
    print(f"   • Subgroup analysis by language")
    print(f"   • Robust hypothesis testing")
    
    start_time = time.time()
    
    results = analyze_multiple_repositories(collector, repositories)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ 200-repository analysis completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Save results
    if results:
        json_file, csv_file = save_results(results)
        
        print(f"\n📁 Regression Analysis Files Created:")
        print(f"   📄 {json_file} (detailed data)")
        print(f"   📊 {csv_file} (regression-ready dataset)")
        
        print(f"\n🎯 Regression Analysis Next Steps:")
        print(f"   1. Load {csv_file} into R, Python, or SPSS")
        print(f"   2. Perform multiple regression: health_score ~ contributors + language + ...")
        print(f"   3. Test interaction effects between variables")
        print(f"   4. Build predictive models for repository health")
        print(f"   5. Validate models with cross-validation")
        
    else:
        print("❌ No results collected. Check for errors above.")

if __name__ == "__main__":
    # Import numpy for log transformations
    import numpy as np
    main()