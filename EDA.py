import pandas as pd

#DATAFRAMES 
dfWeeklyPlayerStatsDefense = pd.read_csv("weekly_player_stats_defense.csv")
dfWeeklyPlayerStatsOffense = pd.read_csv("weekly_player_stats_offense.csv")
dfWeeklyTeamStatsDefense = pd.read_csv("weekly_team_stats_defense.csv")
dfWeeklyTeamStatsOffense = pd.read_csv("weekly_team_stats_offense.csv")
dfYearlyPlayerStatsDefense = pd.read_csv("yearly_player_stats_defense.csv")
dfYearlyPlayerStatsOffense = pd.read_csv("yearly_player_stats_offense.csv")
dfYearlyTeamStatsDefense = pd.read_csv("yearly_team_stats_defense.csv")
dfYearlyTeamStatsOffense = pd.read_csv("yearly_team_stats_offense.csv")

# Show columns for each DataFrame
#print("Weekly Player Stats - Defense Columns:")
#print(dfWeeklyPlayerStatsDefense.columns)

#print("\nWeekly Player Stats - Offense Columns:")
#print(dfWeeklyPlayerStatsOffense.columns)

#print("\nWeekly Team Stats - Defense Columns:")
#print(dfWeeklyTeamStatsDefense.columns)

#print("\nWeekly Team Stats - Offense Columns:")
#print(dfWeeklyTeamStatsOffense.columns)

#print("\nYearly Player Stats - Defense Columns:")
#print(dfYearlyPlayerStatsDefense.columns)

#print("\nYearly Player Stats - Offense Columns:")
#print(dfYearlyPlayerStatsOffense.columns)

#print("\nYearly Team Stats - Defense Columns:")
#print(dfYearlyTeamStatsDefense.columns)

#print("\nYearly Team Stats - Offense Columns:")
#print(dfYearlyTeamStatsOffense.columns)

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# === LOAD & FILTER DATA ===
df = pd.read_csv("yearly_player_stats_offense.csv")

print(df.info())

# === SELECT RELEVANT FEATURES ===
columns_to_use = [
    'player_id', 'player_name', 'season', 'position',
    'season_fantasy_points_ppr',
    'season_pass_touchdown', 'season_rush_touchdown', 'season_receiving_touchdown',
    'season_passing_yards', 'season_rushing_yards', 'season_receiving_yards',
    'season_targets', 'season_receptions', 'games_played_season'
]

df = df[columns_to_use]

# === SORT AND SHIFT TARGET (NEXT YEAR'S POINTS) ===
df = df.sort_values(['player_id', 'season'])
df['next_year_points'] = df.groupby('player_id')['season_fantasy_points_ppr'].shift(-1)
df = df.dropna(subset=['next_year_points'])

# === FEATURE ENGINEERING ===
# Normalize per game
df['ppg'] = df['season_fantasy_points_ppr'] / df['games_played_season']
df['passing_yards_pg'] = df['season_passing_yards'] / df['games_played_season']
df['rushing_yards_pg'] = df['season_rushing_yards'] / df['games_played_season']
df['receiving_yards_pg'] = df['season_receiving_yards'] / df['games_played_season']
df['targets_pg'] = df['season_targets'] / df['games_played_season']
df['receptions_pg'] = df['season_receptions'] / df['games_played_season']

# Encode position
df = pd.get_dummies(df, columns=['position'])

# Drop unneeded columns
df_model = df.drop(columns=[
    'player_id', 'player_name', 'season', 'season_fantasy_points_ppr',
    'season_pass_touchdown', 'season_rush_touchdown', 'season_receiving_touchdown',
    'season_passing_yards', 'season_rushing_yards', 'season_receiving_yards',
    'season_targets', 'season_receptions', 'games_played_season'
])

# === SPLIT & TRAIN MODEL ===
X = df_model.drop(columns=['next_year_points'])
y = df_model['next_year_points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)

# === EVALUATE MODEL ===
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse
