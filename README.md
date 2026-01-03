# Dynamic Player-Based Match Outcome Model
[Premier League Match Predictor](https://fbpredict-82ykzoy2uqpavbxsvzshj5.streamlit.app/)
## Project Description
Traditional football prediction models focus on team-level statistics such as win rates, goals scored, goals conceded, league position. While these approaches capture historical performance, they ignore who will actually step on the pitch and how each player is performing recently.

But match outcomes depend heavily on player availability, form, interactions. For example:
* Removing one playmaker changes expected goals (xG) value drastically.
* A striker’s recent finishing form affects expected conversion.
* Changing center-back pairings influences expected goals against (xGA).

This project aims to build a player-centric prediction system that updates match expectations based on the specific lineup, player form, and how players combine together.

## Data Source and Data Acquisition
In this project, we use the football match data from public source, FBREF, to get the players' stat of the match between 2017 and 2025. The data before 2017 has different data structures, so we cannot use it.

About the data acquisition, we use selenium (Python Library) of automate the web browser to pull out the data from FBREF.

Additionally, we use the github action and github workflow to do the weekly data acquisition of match data.

[FBREF](https://fbref.com/en/)


## Feature Selection
For feature selection, since we want to focus on the player-centric prediction system, we decided to use the player statistics of specific position. For instance, expected goals (xG) and shots for attackers, tackles and blocks for defenders, etc. Additionally, we also put the team features like total shots and key passes.


## Model Training and Result
We used the XGBoost model to train the acquired data, and we used each match data to get the features as we mentioned above to train the model, in order to predict the outcome of the match. However, the quality of the model is relatively not good, the R-squared is just 0.2026, which is really low. We think that because in football matches, there is alot of fluctuation or error which can affect the outcome. For instance, the mental state of players, weathers, fan atmosphere. We could say that somes can be used for prediction, but somes are extremely hard to get, that is the reason why we saw during the match that this team has the winning possibilities of 90%, but still lose.

## Match Forecasting
In the prediction, we will receive the input from the users by dropdown lists of player in each position. After that, we will use the averaged stat of the selected players from the application, and convert them to the features to be the input of the predictions.

![alt text](https://raw.githubusercontent.com/WasuWata/fb_predict/main/photo/dropdown.png)

## Conclusion
To be concluded, there are a open source data of football matches all over the place and we can use them for training, prediction, and create it as applicatioin, however, the data and the features are not enough. For the amount of data, we have just from 2017 to the present, it is not enough. Furthermore, current features cannot explain all the reasons for the match outcomes, there are still a plenty of external factors, such as environment, mental state, and many other things.