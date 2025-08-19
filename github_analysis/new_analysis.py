import pandas as pd
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('repos_updated.csv')

# H1: Compare engagement vs popularity metrics
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Model 1: Popularity metrics
X_pop = df[['log_stars', 'forks']].fillna(0)
X_pop_scaled = scaler.fit_transform(X_pop)
y = df['recent_resolution_rate'].fillna(0)

model_pop = LinearRegression()
model_pop.fit(X_pop_scaled, y)
r2_pop = r2_score(y, model_pop.predict(X_pop_scaled))

# Model 2: Engagement metrics  
X_eng = df[['log_recent_contributors', 'avg_comments_per_issue']].fillna(0)
X_eng_scaled = scaler.fit_transform(X_eng)

model_eng = LinearRegression()
model_eng.fit(X_eng_scaled, y)
r2_eng = r2_score(y, model_eng.predict(X_eng_scaled))

print(f"H1 Results:")
print(f"Popularity model R²: {r2_pop:.3f}")
print(f"Engagement model R²: {r2_eng:.3f}")
print(f"Engagement model explains {(r2_eng/r2_pop - 1)*100:.1f}% more variance")

# H2: Language effects
languages_grouped = df.groupby('language')['health_score'].apply(list)
languages_with_enough_data = [vals for vals in languages_grouped.values if len(vals) >= 5]
h_stat, p_val = stats.kruskal(*languages_with_enough_data)
print(f"\nH2 Results:")
print(f"Kruskal-Wallis H: {h_stat:.2f}, p={p_val:.4f}")

# H5: Already shown but formalize
corr, p = stats.spearmanr(df['stars'].fillna(0), df['recent_resolution_rate'].fillna(0))
print(f"\nH5 Results:")
print(f"Stars vs Resolution: r={corr:.3f}, p={p:.4f}")