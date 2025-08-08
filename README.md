# NBA-player-stat-predictor
A project where users can enter the name of any player that has played a game in the 2024-2025 NBA Regular Season and a sample opponent, and a model will predict their Points, Rebounds, and Assists for that game.

# Data
- The data used for this project is taken from a kaggle dataset which provides individual player statistics for each player in each game for the 2024-2025 NBA Season.
- Before putting this data in code, I used PostgreSQL to manipulate and group together columns
- After this, I used jupyter notebook and used the pandas library to clean the data and prepare rolling averages for each player

# Model
- Used a train-test split method and fed the training data into a regression random forest model and optimized the mean absolute error through a GridSearchCV through 18 combinations

# Opponent Encoding
- encoded opponent data using pandas aswell to group player performance based on opponent difficulty.

# Vizualizaiton
- sperately used matplotlib and seaborn in jupyter notebook to show the similarities between the predicted points and the actual points from the training data

  
![Image](https://github.com/user-attachments/assets/6da41622-d21c-4760-8146-6379dbbcb9b4)
