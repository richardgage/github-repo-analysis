# Dataset Column Guide

## Basic Repository Info
* `repository` - The name of the GitHub project (e.g., "facebook/react")
* `stars` - How many people "starred" it (like a "like" button)
* `forks` - How many people copied the project to work on it
* `language` - What programming language it's mainly written in

## Health & Quality Scores
* `health_score` - Overall project health rating (0-100, higher = healthier)
* `open_ratio` - % of recent issues that are still unresolved (0 = all fixed, 1 = none fixed)

## People Working on the Project
* `total_contributors` - How many different people have contributed code
* `core_contributors` - How many people do most of the work (the "main team")
* `contribution_gini` - How evenly work is distributed (0 = equal, 1 = one person does everything)

## Development Activity
* `total_commits` - How many code changes were made (in last 6 months)
* `commits_per_day` - Average code changes per day
* `activity_ratio` - What % of days had any activity (0.5 = active half the days)
* `unique_authors` - How many different people made code changes

## Issue Management (Bug Reports & Requests)
* `open_issues` - Current number of unresolved problems/requests
* `recent_issues_count` - New issues created in last 6 months
* `recent_issues_per_day` - How many new issues appear daily
* `recent_resolution_rate` - % of recent issues that got solved
* `avg_comments_per_issue` - How much discussion each issue gets

## Helper Variables for Analysis
* `log_stars` - Mathematical transformation of stars (for better statistics)
* `log_contributors` - Mathematical transformation of contributors
* `contributors_per_star` - How many contributors relative to popularity
* `commits_per_contributor` - How productive each contributor is
* `time_window_days` - Time period analyzed (180 = 6 months)


**Ways to Think of Them**
- **Stars/Forks** = How popular/famous it is
- **Contributors** = Size of the team
- **Commits** = How much work gets done
- **Issues** = Customer service quality
- **Health Score** = Overall business health