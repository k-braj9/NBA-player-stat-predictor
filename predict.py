import pandas as pd
from unidecode import unidecode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("PlayerStatistics.csv")

# Normalize names
data['Name'] = data['firstName'].str.lower().apply(unidecode)

# Get user input
player_name = input("Enter NBA player name (e.x, LeBron James): ")
opponent_input = input("Enter Opp Team Name (e.x, Warriors): ")

player = unidecode(player_name.lower())
player_data = data[data['Name'] == player]
print(player_data)

if player_data.empty:
    print("Player does not exist or has not played a game")

# Encode opponent team names
le = LabelEncoder()
data['Opponent'] = le.fit_transform(data['opponentteamName'])

# Compute rolling stats
stats = ['MP','FGA','FG%','3PA','3P%','FTA','FT%','AST','TRB','TOV','PTS']
for stat in stats:
    data[f'new {stat}'] = data.groupby('Name')[stat].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

player = unidecode(player_name.lower())
player_data = data[data['Name'] == player]

# Feature and target selection
new_data = [f'new {col}' for col in stats] + ['Opponent']
X = data[new_data]
y = data[['PTS', 'AST', 'TRB']]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluation
print("Model Evaluation:")
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
print("R2:", r2_score(y_test, model.predict(X_test)))

# Encode opponent for prediction
chosen_team = None
for team in le.classes_:
    if opponent_input.lower() == team.lower():
        chosen_team = team
        break

opponent_encoded = le.transform([chosen_team])[0]

# Prepare prediction input
last_game = player_data.tail(1)[new_data]
last_game['Opponent'] = opponent_encoded

# Prediction
prediction = model.predict(last_game)[0]

# Output
print(f"Predicted Next Game Statline vs {opponent_input}:")
print(f"PTS: {int(round(prediction[0]))}")
print(f"AST: {int(round(prediction[1]))}")
print(f"TRB: {int(round(prediction[2]))}")

