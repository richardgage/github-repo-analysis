#!/usr/bin/env python3
"""
Linear Regression Analysis for GitHub Repository Resolution Rates
================================================================

This script performs comprehensive linear regression analysis to predict
recent_resolution_rate using various repository characteristics.

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    """Load and clean the GitHub repository dataset."""
    print("Loading and cleaning data...")
    
    # Load data
    df = pd.read_csv(filepath)
    
    if 'health_score' in df.columns:
        df = df.drop('health_score', axis=1)
    
    # Filter for valid data
    initial_rows = len(df)
    df = df.dropna(subset=['recent_resolution_rate'])
    df = df[df['stars'] > 0]
    if 'recent_contributors' in df.columns:
        df = df[df['recent_contributors'] > 0]
    else:
        df = df[df['total_contributors'] > 0]
    df = df[df['recent_resolution_rate'].between(0, 1)]
    
    print(f"Data loaded: {initial_rows} -> {len(df)} rows after cleaning")
    print(f"Columns: {list(df.columns)}")
    
    return df

def exploratory_analysis(df):
    """Perform exploratory data analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print("\nTarget Variable (recent_resolution_rate) Statistics:")
    print(f"Mean: {df['recent_resolution_rate'].mean():.4f}")
    print(f"Std:  {df['recent_resolution_rate'].std():.4f}")
    print(f"Min:  {df['recent_resolution_rate'].min():.4f}")
    print(f"Max:  {df['recent_resolution_rate'].max():.4f}")
    
    # Programming language distribution
    if 'language' in df.columns:
        print(f"\nProgramming Language Distribution:")
        lang_counts = df['language'].value_counts().head(10)
        for lang, count in lang_counts.items():
            avg_resolution = df[df['language'] == lang]['recent_resolution_rate'].mean()
            print(f"{lang}: {count} repos, {avg_resolution:.3f} avg resolution rate")

def prepare_features(df):
    """Prepare features for regression analysis."""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)
    
    # Define numeric features for regression
    numeric_features = [
        'log_stars',
        'recent_contributors',
        'log_recent_contributors',
        'recent_contributor_ratio',
        'open_ratio', 'commits_per_day',
        'avg_comments_per_issue', 'activity_ratio', 'contribution_gini',
        'stars', 'forks', 'open_issues'
    ]
    
    # Keep only features that exist in the dataset
    available_features = [f for f in numeric_features if f in df.columns]
    print(f"Available numeric features: {available_features}")
    
    # Create feature matrix
    X = df[available_features].copy()
    y = df['recent_resolution_rate'].copy()

    # Replace inf/-inf with NaN, then fill or drop
    X = X.replace([np.inf, -np.inf], np.nan)
    # Option 1: Drop rows with NaN (safer for regression)
    valid_idx = ~X.isnull().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Add language dummy variables if language column exists
    if 'language' in df.columns:
        # Get top 5 languages
        top_languages = df['language'].value_counts().head(5).index
        for lang in top_languages:
            X[f'is_{lang.lower().replace("+", "plus").replace("#", "sharp")}'] = (df['language'] == lang).astype(int)
        print(f"Added language dummy variables for: {list(top_languages)}")
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y

def simple_linear_regressions(X, y):
    """Perform simple linear regression for each feature."""
    print("\n" + "="*60)
    print("SIMPLE LINEAR REGRESSIONS")
    print("="*60)
    
    results = []
    
    for feature in X.columns:
        if X[feature].dtype in ['int64', 'float64']:  # Only numeric features
            # Prepare data
            x_simple = X[feature].values.reshape(-1, 1)
            
            # Fit model
            model = LinearRegression()
            model.fit(x_simple, y)
            
            # Calculate metrics
            y_pred = model.predict(x_simple)
            r2 = r2_score(y, y_pred)
            
            # Calculate correlation and p-value
            correlation, p_value = stats.pearsonr(X[feature], y)
            
            results.append({
                'feature': feature,
                'coefficient': model.coef_[0],
                'intercept': model.intercept_,
                'r2': r2,
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # Sort by R-squared
    results = sorted(results, key=lambda x: x['r2'], reverse=True)
    
    print("Simple Linear Regression Results (sorted by R²):")
    print(f"{'Feature':<25} {'Coefficient':<12} {'R²':<8} {'Correlation':<12} {'P-value':<10} {'Significant'}")
    print("-" * 85)
    
    for result in results:
        sig_marker = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['significant'] else ""
        print(f"{result['feature']:<25} {result['coefficient']:<12.4f} {result['r2']:<8.4f} {result['correlation']:<12.4f} {result['p_value']:<10.4f} {sig_marker}")
    
    return results

def multiple_linear_regression(X, y):
    """Perform multiple linear regression with different approaches."""
    print("\n" + "="*60)
    print("MULTIPLE LINEAR REGRESSION")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Different regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression (α=1.0)': Ridge(alpha=1.0),
        'Lasso Regression (α=0.1)': Lasso(alpha=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"Training R²:   {train_r2:.4f}")
        print(f"Testing R²:    {test_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE:  {test_rmse:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"CV R² Mean:    {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # Feature importance (coefficients)
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:<25}: {row['coefficient']:>8.4f}")
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'feature_importance': feature_importance if hasattr(model, 'coef_') else None
        }
    
    return results, scaler, X_train, X_test, y_train, y_test

def statistical_tests(X, y):
    """Perform statistical significance tests for regression."""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*60)
    
    # Full model regression for statistical tests
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Calculate coefficients using normal equation
    try:
        coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        # Predictions and residuals
        y_pred = X_with_intercept @ coefficients
        residuals = y - y_pred
        
        # Calculate standard errors
        mse = np.sum(residuals**2) / (len(y) - len(coefficients))
        covariance_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        standard_errors = np.sqrt(np.diag(covariance_matrix))
        
        # T-statistics and p-values
        t_stats = coefficients / standard_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coefficients)))
        
        # Create results table
        feature_names = ['Intercept'] + list(X.columns)
        
        print("Coefficient Significance Tests:")
        print(f"{'Feature':<25} {'Coefficient':<12} {'Std Error':<12} {'t-stat':<10} {'p-value':<10} {'Significant'}")
        print("-" * 85)
        
        for i, feature in enumerate(feature_names):
            sig_marker = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
            print(f"{feature:<25} {coefficients[i]:<12.4f} {standard_errors[i]:<12.4f} {t_stats[i]:<10.3f} {p_values[i]:<10.4f} {sig_marker}")
        
        # Model statistics
        r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
        adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(coefficients))
        f_stat = (r2 / (len(coefficients) - 1)) / ((1 - r2) / (len(y) - len(coefficients)))
        
        print(f"\nModel Statistics:")
        print(f"R²:           {r2:.4f}")
        print(f"Adjusted R²:  {adjusted_r2:.4f}")
        print(f"F-statistic:  {f_stat:.4f}")
        print(f"RMSE:         {np.sqrt(mse):.4f}")
        
    except np.linalg.LinAlgError:
        print("Error: Matrix inversion failed. Check for multicollinearity.")

def language_specific_analysis(df):
    """Analyze resolution rates by programming language."""
    print("\n" + "="*60)
    print("PROGRAMMING LANGUAGE ANALYSIS")
    print("="*60)
    
    if 'language' not in df.columns:
        print("Language column not found in dataset.")
        return
    
    # Get languages with at least 5 repositories
    lang_counts = df['language'].value_counts()
    languages_to_analyze = lang_counts[lang_counts >= 5].index
    
    print("Resolution Rates by Programming Language:")
    print(f"{'Language':<15} {'Count':<8} {'Mean':<8} {'Std':<8} {'Median':<8}")
    print("-" * 55)
    
    lang_stats = []
    for lang in languages_to_analyze:
        lang_data = df[df['language'] == lang]['recent_resolution_rate']
        stats_dict = {
            'language': lang,
            'count': len(lang_data),
            'mean': lang_data.mean(),
            'std': lang_data.std(),
            'median': lang_data.median()
        }
        lang_stats.append(stats_dict)
        
        print(f"{lang:<15} {stats_dict['count']:<8} {stats_dict['mean']:<8.3f} {stats_dict['std']:<8.3f} {stats_dict['median']:<8.3f}")
    
    # ANOVA test
    groups = [df[df['language'] == lang]['recent_resolution_rate'] for lang in languages_to_analyze]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\nANOVA Test Results:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant difference between languages: {'Yes' if p_value < 0.05 else 'No'}")
    
    return lang_stats

def create_visualizations(df, X, y, simple_results, multiple_results, scaler):
    """Create comprehensive visualizations for the analysis."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Distribution of target variable
    plt.subplot(3, 4, 1)
    plt.hist(y, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Resolution Rate', fontsize=12, fontweight='bold')
    plt.xlabel('Recent Resolution Rate')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Box plot by language (if available)
    if 'language' in df.columns:
        plt.subplot(3, 4, 2)
        top_languages = df['language'].value_counts().head(6).index
        df_lang = df[df['language'].isin(top_languages)]
        sns.boxplot(data=df_lang, x='language', y='recent_resolution_rate')
        plt.title('Resolution Rate by Language', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # 3. Correlation heatmap
    plt.subplot(3, 4, 3)
    numeric_cols = X.select_dtypes(include=[np.number]).columns[:8]  # Top 8 numeric features
    corr_matrix = X[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    # 4. Top predictors from simple regression
    plt.subplot(3, 4, 4)
    top_features = [r['feature'] for r in simple_results[:6]]
    r2_scores = [r['r2'] for r in simple_results[:6]]
    bars = plt.bar(range(len(top_features)), r2_scores, alpha=0.7)
    plt.title('Top Predictors (R² Score)', fontsize=12, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('R² Score')
    plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5-8. Scatter plots for top 4 predictors
    for i, result in enumerate(simple_results[:4]):
        plt.subplot(3, 4, 5+i)
        feature = result['feature']
        if feature in X.columns:
            plt.scatter(X[feature], y, alpha=0.6, s=30)
            
            # Add regression line
            x_range = np.linspace(X[feature].min(), X[feature].max(), 100)
            y_pred = result['coefficient'] * x_range + result['intercept']
            plt.plot(x_range, y_pred, 'r-', linewidth=2)
            
            plt.title(f'{feature}\n(R²={result["r2"]:.3f})', fontsize=11, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Resolution Rate')
            plt.grid(True, alpha=0.3)
    
    # 9. Residual plot for best model
    plt.subplot(3, 4, 9)
    best_model = multiple_results['Linear Regression']['model']
    X_scaled = scaler.transform(X)
    y_pred = best_model.predict(X_scaled)
    residuals = y - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Residual Plot', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    # 10. Actual vs Predicted
    plt.subplot(3, 4, 10)
    plt.scatter(y, y_pred, alpha=0.6, s=30)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title('Actual vs Predicted', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Resolution Rate')
    plt.ylabel('Predicted Resolution Rate')
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = multiple_results['Linear Regression']['test_r2']
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 11. Feature importance comparison
    plt.subplot(3, 4, 11)
    models_to_compare = ['Linear Regression', 'Ridge Regression (α=1.0)', 'Lasso Regression (α=0.1)']
    r2_scores = [multiple_results[model]['test_r2'] for model in models_to_compare]
    model_names = ['Linear', 'Ridge', 'Lasso']
    
    bars = plt.bar(model_names, r2_scores, alpha=0.7, color=['blue', 'green', 'orange'])
    plt.title('Model Comparison (Test R²)', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score')
    plt.ylim(0, max(r2_scores) * 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 12. Stars vs Resolution Rate (log scale)
    plt.subplot(3, 4, 12)
    if 'log_stars' in X.columns:
        plt.scatter(X['log_stars'], y, alpha=0.6, s=30)
        plt.title('Log Stars vs Resolution Rate', fontsize=12, fontweight='bold')
        plt.xlabel('Log Stars')
        plt.ylabel('Resolution Rate')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(X['log_stars'], y, 1)
        p = np.poly1d(z)
        plt.plot(X['log_stars'], p(X['log_stars']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('regression_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate detailed feature importance plot
    plt.figure(figsize=(12, 8))
    
    # Get feature importance from Linear Regression
    feature_importance = multiple_results['Linear Regression']['feature_importance']
    if feature_importance is not None:
        # Plot top 15 features
        top_15 = feature_importance.head(15)
        
        plt.subplot(2, 1, 1)
        colors = ['red' if coef < 0 else 'blue' for coef in top_15['coefficient']]
        bars = plt.barh(range(len(top_15)), top_15['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_15)), top_15['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, coef) in enumerate(zip(bars, top_15['coefficient'])):
            plt.text(coef + (0.01 if coef >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{coef:.3f}', ha='left' if coef >= 0 else 'right', va='center', fontsize=9)
    
    # Model performance comparison with error bars
    plt.subplot(2, 1, 2)
    models = ['Linear Regression', 'Ridge Regression (α=1.0)', 'Lasso Regression (α=0.1)']
    train_scores = [multiple_results[model]['train_r2'] for model in models]
    test_scores = [multiple_results[model]['test_r2'] for model in models]
    cv_scores = [multiple_results[model]['cv_mean'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, train_scores, width, label='Training R²', alpha=0.8, color='lightblue')
    plt.bar(x, test_scores, width, label='Testing R²', alpha=0.8, color='lightcoral')
    plt.bar(x + width, cv_scores, width, label='CV R²', alpha=0.8, color='lightgreen')
    
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, ['Linear', 'Ridge', 'Lasso'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(train_scores), max(test_scores), max(cv_scores)) * 1.1)
    
    plt.tight_layout()
    plt.savefig('feature_importance_and_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Plots saved as 'regression_analysis_plots.png' and 'feature_importance_and_model_comparison.png'")

def create_prediction_example(best_model, scaler, X):
    """Create example predictions for different repository types."""
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES")
    print("="*60)
    
    # Create example repositories
    examples = {
        'High-Star JS Project': {
            'log_stars': 11.0,  # ~60k stars
            'total_contributors': 400,
            'open_ratio': 0.3,
            'commits_per_day': 2.0,
            'avg_comments_per_issue': 3.0,
            'activity_ratio': 0.8
        },
        'Small Python Project': {
            'log_stars': 8.0,  # ~3k stars
            'total_contributors': 50,
            'open_ratio': 0.6,
            'commits_per_day': 0.5,
            'avg_comments_per_issue': 2.0,
            'activity_ratio': 0.3
        },
        'Enterprise Java Project': {
            'log_stars': 9.5,  # ~13k stars
            'total_contributors': 200,
            'open_ratio': 0.2,
            'commits_per_day': 1.5,
            'avg_comments_per_issue': 4.0,
            'activity_ratio': 0.6
        }
    }
    
    print("Predicted Resolution Rates for Example Repositories:")
    print("-" * 55)
    
    for name, features in examples.items():
        # Create feature vector (only using features that exist in model)
        feature_vector = []
        for col in X.columns:
            if col in features:
                feature_vector.append(features[col])
            elif col.startswith('is_'):
                # Language dummy variables
                lang = col.replace('is_', '').replace('plus', '+').replace('sharp', '#')
                feature_vector.append(1 if lang.lower() in name.lower() else 0)
            else:
                # Use median value for missing features
                feature_vector.append(X[col].median())
        
        # Make prediction
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)
        prediction = best_model.predict(feature_vector_scaled)[0]
        
        print(f"{name}: {prediction:.3f} ({prediction*100:.1f}%)")

def main():
    """Main analysis function."""
    print("GitHub Repository Resolution Rate Analysis")
    print("=" * 60)
    
    # Update this path to your CSV file
    filepath = "repos_filtered.csv"
    
    try:
        # Load and clean data
        df = load_and_clean_data(filepath)
        
        # Exploratory analysis
        exploratory_analysis(df)
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Simple linear regressions
        simple_results = simple_linear_regressions(X, y)
        
        # Multiple linear regression
        multiple_results, scaler, X_train, X_test, y_train, y_test = multiple_linear_regression(X, y)
        
        # Statistical tests
        statistical_tests(X, y)
        
        # Language analysis
        language_specific_analysis(df)
        
        # Create visualizations
        create_visualizations(df, X, y, simple_results, multiple_results, scaler)
        
        # Prediction examples
        best_model = multiple_results['Linear Regression']['model']
        create_prediction_example(best_model, scaler, X)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Key findings:")
        print("1. Check simple regression results for strongest individual predictors")
        print("2. Compare multiple regression models for best performance")
        print("3. Review statistical significance tests for reliable predictors")
        print("4. Examine language-specific patterns for ecosystem insights")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{filepath}'")
        print("Please update the filepath variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"Error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()