import pandas as pd
from unidecode import unidecode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("PlayerStatistics.csv")

# Normalize names
data['name'] = data['firstName'].str.lower().apply(unidecode)

# Get user input
player_name = input("Enter NBA player name: ")
opponent_input = input("Enter Opp Team Name: ")

player = unidecode(player_name.lower())

# Encode opponent team names
le = LabelEncoder()
data['opponent'] = le.fit_transform(data['opponentteamName'])

# Compute rolling stats
stats = ['MP','FGA','FG%','3PA','3P%','FTA','FT%','AST','TRB','TOV','PTS']
for stat in stats:
    data[f'new {stat}'] = data.groupby('name')[stat].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

player_data = data[data['name'] == player]
print(player_data)

if player_data.empty:
    print("Player does not exist or has not played a game")
    
# Feature and target selection
new_data = [f'new {col}' for col in stats] + ['opponent']
X = data[new_data]
y = data[['PTS', 'AST', 'TRB']]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluation
print("Model Evaluation:")
X_test = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, X_test))
print("r2:", r2_score(y_test, X_test))

# Encode opponent for prediction
chosen_team = None
for team in data['opponentteamName']:
    if opponent_input == team:
        chosen_team = team

chosen_opponent = le.transform([chosen_team])[0]

# Prepare prediction input
last_game = player_data.tail(1)[new_data].copy()
last_game['opponent'] = chosen_opponent

# Prediction
prediction = model.predict(last_game)[0]
points = int(round(prediction[0]))
assists = int(round(prediction[1]))
rebounds = int(round(prediction[1]))
# Output
print(f'Predicted Next Game Statline vs {opponent_input}:')
print(f'{points} points')
print(f'{assists} assists')
print(f'{rebounds} rebounds')

