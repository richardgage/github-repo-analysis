#!/usr/bin/env python3
"""
Interactive Plotting Script for GitHub Repository Analysis
==========================================================

This script creates additional interactive plots and allows you to explore
specific relationships in your data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def create_advanced_plots(filepath="dataset.csv"):
    """Create advanced plots for deeper analysis."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Remove health_score if present
    if 'health_score' in df.columns:
        df = df.drop('health_score', axis=1)
    
    # Clean data
    df = df.dropna(subset=['recent_resolution_rate'])
    df = df[df['stars'] > 0]
    df = df[df['total_contributors'] > 0]
    df = df[df['recent_resolution_rate'].between(0, 1)]
    
    print(f"Data cleaned: {len(df)} repositories")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Stars vs Resolution Rate (with trend lines)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['log_stars'], df['recent_resolution_rate'], 
                         c=df['total_contributors'], alpha=0.6, s=50, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(df['log_stars'], df['recent_resolution_rate'], 1)
    p = np.poly1d(z)
    ax1.plot(df['log_stars'], p(df['log_stars']), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Log Stars')
    ax1.set_ylabel('Resolution Rate')
    ax1.set_title('Stars vs Resolution Rate\n(Color = Contributors)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Total Contributors')
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(df['log_stars'], df['recent_resolution_rate'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # 2. Open Ratio vs Resolution Rate by Language
    ax2 = axes[0, 1]
    top_langs = df['language'].value_counts().head(5).index
    colors = sns.color_palette("Set2", len(top_langs))
    
    for i, lang in enumerate(top_langs):
        lang_data = df[df['language'] == lang]
        ax2.scatter(lang_data['open_ratio'], lang_data['recent_resolution_rate'], 
                   label=lang, alpha=0.7, s=50, color=colors[i])
    
    ax2.set_xlabel('Open Ratio')
    ax2.set_ylabel('Resolution Rate')
    ax2.set_title('Open Ratio vs Resolution Rate\nby Programming Language', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Contributors vs Activity Patterns
    ax3 = axes[0, 2]
    
    # Create size categories
    df['repo_size'] = pd.cut(df['total_contributors'], 
                            bins=[0, 10, 50, 200, float('inf')], 
                            labels=['Small (≤10)', 'Medium (11-50)', 'Large (51-200)', 'Huge (200+)'])
    
    sns.boxplot(data=df, x='repo_size', y='recent_resolution_rate', ax=ax3)
    ax3.set_xlabel('Repository Size (by Contributors)')
    ax3.set_ylabel('Resolution Rate')
    ax3.set_title('Resolution Rate by Repository Size', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Commit Activity vs Resolution Patterns
    ax4 = axes[1, 0]
    
    # Create activity bins
    df['activity_level'] = pd.cut(df['commits_per_day'], 
                                 bins=[0, 0.5, 2, 10, float('inf')], 
                                 labels=['Low (≤0.5)', 'Medium (0.5-2)', 'High (2-10)', 'Very High (10+)'])
    
    activity_stats = df.groupby('activity_level')['recent_resolution_rate'].agg(['mean', 'std', 'count'])
    
    bars = ax4.bar(range(len(activity_stats)), activity_stats['mean'], 
                   yerr=activity_stats['std'], capsize=5, alpha=0.7)
    ax4.set_xticks(range(len(activity_stats)))
    ax4.set_xticklabels(activity_stats.index, rotation=45)
    ax4.set_xlabel('Daily Commit Activity Level')
    ax4.set_ylabel('Mean Resolution Rate')
    ax4.set_title('Resolution Rate by Activity Level\n(Error bars = Std Dev)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, activity_stats['count'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 5. Comments vs Resolution Success
    ax5 = axes[1, 1]
    
    # Bin comments for better visualization
    df['comment_level'] = pd.cut(df['avg_comments_per_issue'], 
                                bins=[0, 1, 3, 10, float('inf')], 
                                labels=['Few (≤1)', 'Some (1-3)', 'Many (3-10)', 'Lots (10+)'])
    
    # Create violin plot
    sns.violinplot(data=df, x='comment_level', y='recent_resolution_rate', ax=ax5)
    ax5.set_xlabel('Comments per Issue Level')
    ax5.set_ylabel('Resolution Rate')
    ax5.set_title('Resolution Rate Distribution\nby Comment Activity', fontweight='bold')
    
    # 6. Multi-dimensional scatter plot
    ax6 = axes[1, 2]
    
    # Size by stars, color by language (top 4 only)
    top_4_langs = df['language'].value_counts().head(4).index
    df_filtered = df[df['language'].isin(top_4_langs)]
    
    for lang in top_4_langs:
        lang_data = df_filtered[df_filtered['language'] == lang]
        sizes = np.sqrt(lang_data['stars']) / 10  # Scale for visibility
        ax6.scatter(lang_data['contribution_gini'], lang_data['recent_resolution_rate'], 
                   s=sizes, alpha=0.6, label=lang)
    
    ax6.set_xlabel('Contribution Inequality (Gini)')
    ax6.set_ylabel('Resolution Rate')
    ax6.set_title('Multi-dimensional View\n(Size = Stars, Color = Language)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation matrix heatmap
    plt.figure(figsize=(12, 10))
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    
    plt.title('Correlation Matrix of Repository Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics by language
    print("\n" + "="*60)
    print("DETAILED LANGUAGE ANALYSIS")
    print("="*60)
    
    lang_summary = df.groupby('language').agg({
        'recent_resolution_rate': ['count', 'mean', 'std', 'median'],
        'avg_comments_per_issue': 'mean',
        'open_ratio': 'mean',
        'total_contributors': 'mean',
        'log_stars': 'mean'
    }).round(3)
    
    print(lang_summary)
    
    print(f"\n✓ Advanced plots saved as:")
    print(f"  - 'advanced_analysis_plots.png'")
    print(f"  - 'correlation_heatmap.png'")

def create_custom_scatter_plot(x_col, y_col, color_col=None, size_col=None, 
                              filepath="github_analysis_200_20250728_124301.csv"):
    """Create a custom scatter plot with specified columns."""
    
    df = pd.read_csv(filepath)
    if 'health_score' in df.columns:
        df = df.drop('health_score', axis=1)
    
    df = df.dropna(subset=[x_col, y_col])
    
    plt.figure(figsize=(10, 8))
    
    if color_col and size_col:
        scatter = plt.scatter(df[x_col], df[y_col], c=df[color_col], s=df[size_col]/100, 
                            alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label=color_col)
    elif color_col:
        scatter = plt.scatter(df[x_col], df[y_col], c=df[color_col], alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label=color_col)
    elif size_col:
        plt.scatter(df[x_col], df[y_col], s=df[size_col]/100, alpha=0.6)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.6)
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col}', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add correlation
    if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
        corr, p_val = stats.pearsonr(df[x_col], df[y_col])
        plt.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Creating advanced plots...")
    create_advanced_plots()
    
    print("\nYou can also create custom plots using:")
    print("create_custom_scatter_plot('log_stars', 'recent_resolution_rate', 'total_contributors')")
    print("create_custom_scatter_plot('open_ratio', 'recent_resolution_rate', 'language')")
