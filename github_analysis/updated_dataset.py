import pandas as pd
import numpy as np

# Read your CSV
df = pd.read_csv('dataset.csv')

# Create new columns using existing data
df['recent_contributors'] = df['unique_authors']  # These are 180-day contributors
df['log_recent_contributors'] = np.log(df['recent_contributors'] + 1)  # Add 1 to handle zeros

# Calculate recent contributor ratio
df['recent_contributor_ratio'] = df['recent_contributors'] / df['total_contributors']
df['recent_contributor_ratio'] = df['recent_contributor_ratio'].fillna(0)  # Handle division by zero

# Recalculate contributors per star using recent data
df['recent_contributors_per_star'] = df['recent_contributors'] / df['stars']
df['recent_contributors_per_star'] = df['recent_contributors_per_star'].replace([np.inf, -np.inf], 0)

# Save updated CSV
df.to_csv('repos_updated.csv', index=False)

# Print summary statistics to verify
print("Summary of new metrics:")
print(f"Mean recent contributors: {df['recent_contributors'].mean():.1f}")
print(f"Median recent contributors: {df['recent_contributors'].median():.1f}")
print(f"Mean log(recent_contributors): {df['log_recent_contributors'].mean():.2f}")
print(f"This translates to ~{np.exp(df['log_recent_contributors'].mean()):.0f} contributors on average")