import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# === LOAD MAIN DATASET ===
df = pd.read_csv("yearly_player_stats_offense.csv")

# === FILTER & PREPARE COLUMNS ===
cols = [
    'player_id', 'player_name', 'season', 'position',
    'season_fantasy_points_ppr',
    'season_pass_touchdown', 'season_rush_touchdown', 'season_receiving_touchdown',
    'season_passing_yards', 'season_rushing_yards', 'season_receiving_yards',
    'season_targets', 'season_receptions', 'games_played_season'
]
df = df[cols].copy()

# === ADD NEXT YEAR'S TARGET ===
df = df.sort_values(['player_id', 'season'])
df['next_year_points'] = df.groupby('player_id')['season_fantasy_points_ppr'].shift(-1)
df = df.dropna(subset=['next_year_points'])

# === FEATURE ENGINEERING ===
df['ppg'] = df['season_fantasy_points_ppr'] / df['games_played_season']
df['passing_yards_pg'] = df['season_passing_yards'] / df['games_played_season']
df['rushing_yards_pg'] = df['season_rushing_yards'] / df['games_played_season']
df['receiving_yards_pg'] = df['season_receiving_yards'] / df['games_played_season']
df['targets_pg'] = df['season_targets'] / df['games_played_season']
df['receptions_pg'] = df['season_receptions'] / df['games_played_season']
df = pd.get_dummies(df, columns=['position'])

drop_cols = [
    'player_id', 'player_name', 'season', 'season_fantasy_points_ppr',
    'season_pass_touchdown', 'season_rush_touchdown', 'season_receiving_touchdown',
    'season_passing_yards', 'season_rushing_yards', 'season_receiving_yards',
    'season_targets', 'season_receptions', 'games_played_season'
]
df_model = df.drop(columns=drop_cols)

# === MODEL TRAINING ===
X = df_model.drop(columns=['next_year_points'])
y = df_model['next_year_points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)

# === APPLY MODEL TO MOST RECENT SEASON ===
latest_season = df['season'].max() - 1
recent = df[df['season'] == latest_season].copy()
features = recent.drop(columns=drop_cols + ['next_year_points'])
recent['predicted_points'] = model.predict(features)

# === LOAD FANTASY TEAM ===
my_team = pd.read_csv("fantasyTeam.csv")

# === MERGE TEAM DATA WITH PREDICTIONS ===
my_team = pd.merge(my_team, recent, on='player_name', how='left')

# === BUILD TRADE SUGGESTIONS ===
suggestions = []
for _, row in my_team.iterrows():
    pos = row['position']
    my_pred = row['predicted_points']
    try:
        # Use one-hot encoding to filter candidates
        candidates = recent[recent[f'position_{pos}'] == 1]
    except KeyError:
        continue

    upgrades = candidates[candidates['predicted_points'] > my_pred].copy()
    upgrades['upgrade'] = upgrades['predicted_points'] - my_pred
    upgrades = upgrades.sort_values(by='upgrade', ascending=False).head(3)

    for _, up in upgrades.iterrows():
        suggestions.append({
            'Your Player': row['player_name'],
            'Their Player': up['player_name'],
            'Position': pos,
            'Your Predicted Points': round(my_pred, 1),
            'Their Predicted Points': round(up['predicted_points'], 1),
            'Upgrade Value': round(up['upgrade'], 1)
        })

# === DISPLAY RESULTS ===
suggestions_df = pd.DataFrame(suggestions)
print("\nTop Trade Suggestions:")
print(suggestions_df.to_string(index=False))
