import pandas as pd
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv("yearly_player_stats_offense.csv")

# Columns to use
columns_to_use = [
    'player_id', 'player_name', 'season', 'position',
    'season_fantasy_points_ppr',
    'season_pass_touchdown', 'season_rush_touchdown', 'season_receiving_touchdown',
    'season_passing_yards', 'season_rushing_yards', 'season_receiving_yards',
    'season_targets', 'season_receptions', 'games_played_season'
]
df = df[columns_to_use].copy()

# Sort and shift to get next year target
df = df.sort_values(['player_id', 'season'])
df['next_year_points'] = df.groupby('player_id')['season_fantasy_points_ppr'].shift(-1)
df = df.dropna(subset=['next_year_points'])

# Feature engineering
df['ppg'] = df['season_fantasy_points_ppr'] / df['games_played_season']
df['passing_yards_pg'] = df['season_passing_yards'] / df['games_played_season']
df['rushing_yards_pg'] = df['season_rushing_yards'] / df['games_played_season']
df['receiving_yards_pg'] = df['season_receiving_yards'] / df['games_played_season']
df['targets_pg'] = df['season_targets'] / df['games_played_season']
df['receptions_pg'] = df['season_receptions'] / df['games_played_season']
df = pd.get_dummies(df, columns=['position'])

# Drop unused
drop_cols = [
    'player_id', 'player_name', 'season', 'season_fantasy_points_ppr',
    'season_pass_touchdown', 'season_rush_touchdown', 'season_receiving_touchdown',
    'season_passing_yards', 'season_rushing_yards', 'season_receiving_yards',
    'season_targets', 'season_receptions', 'games_played_season'
]
df_model = df.drop(columns=drop_cols)
X = df_model.drop(columns=['next_year_points'])
y = df_model['next_year_points']

# Train model
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X, y)

# Predict most recent season
latest_season = df['season'].max() - 1
recent = df[df['season'] == latest_season].copy()
features = recent.drop(columns=drop_cols + ['next_year_points'])
recent['predicted_points'] = model.predict(features)

# Save trade values
trade_values = recent[['player_name', 'predicted_points']].sort_values(by='predicted_points', ascending=False)
trade_values = trade_values.sort_values(by='predicted_points', ascending=False)
trade_values.to_csv("player_trade_values.csv", index=False)
