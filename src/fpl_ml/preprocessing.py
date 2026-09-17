from sklearn.model_selection import train_test_split
from urllib.request import urlopen
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pyarrow

#base url for all endpoints
base_url = 'https://fantasy.premierleague.com/api/'
#load the data from api
response = urlopen(base_url+'bootstrap-static/')
data = json.loads(response.read())

#players
players_df = pd.DataFrame(data['elements'])
players_df = players_df.loc[players_df['minutes']>0] #Only players who have actually played minutes
players_df = players_df[['id','element_type']]

#fixtures
response = urlopen(base_url+'fixtures/')
data = json.loads(response.read())
fixtures_df = pd.DataFrame(data)
fixtures_df = fixtures_df.loc[fixtures_df['finished']==True] #Only completed fixtures
fixtures_df = fixtures_df[['id', 'team_h_difficulty', 'team_a_difficulty']] #To add a fixture difficulty parameter

#Access the players' historical fixture data - only for players with minutes this season > 0
history_list = []
for id_num in players_df['id']: #Loops through all players with >0 minutes
    response = urlopen(f'{base_url}element-summary/{id_num}/')
    data = json.loads(response.read())
    if len(data['history'])>=3: #Only players with at least three games - gives minimum 2 gws data for predicting
        for i in range(len(data['history'])):
            history_list.append(data['history'][i])

players_data_df = pd.DataFrame(history_list)

#merge players data and fixture info
players_data_df = players_data_df.merge(fixtures_df,how='inner', left_on='fixture', right_on='id')
players_data_df['fdr'] = np.where(players_data_df['was_home'],
                                  players_data_df['team_h_difficulty'],
                                  players_data_df['team_a_difficulty'])
#Merges the home or away difficulty based on whether player was home or away in target fixture

#create a column for the targets - i.e. the next week's points
#Loop through rows
#identify row with 'round' +2 and element_1 == element_2
#If this row exists append the data for the second input week to the row - also add the points for the target week as targets
#Otherwise next_points = nan
target_points_list = []
target_fdr_list= []
target_was_home_list = []

total_points_2_list = []
ict_2_list = []
expected_goal_involvements_2_list = []
expected_goals_conceded_2_list = []
minutes_2_list = []

for _,row in players_data_df.iterrows():
    game_2_row = (players_data_df.loc[(players_data_df['element']==row['element']) & (players_data_df['round']==(row['round']+1))])
    target_row = (players_data_df.loc[(players_data_df['element']==row['element']) & (players_data_df['round']==(row['round']+2))])


    try:
        total_points_2_list.append(game_2_row.iloc[0]['total_points'])
        ict_2_list.append(game_2_row.iloc[0]['ict_index'])
        expected_goal_involvements_2_list.append(game_2_row.iloc[0]['expected_goal_involvements'])
        expected_goals_conceded_2_list.append(game_2_row.iloc[0]['expected_goals_conceded'])
        minutes_2_list.append(game_2_row.iloc[0]['minutes'])
        try:
            target_points_list.append(target_row.iloc[0]['total_points'])
            target_fdr_list.append(target_row.iloc[0]['fdr'])
            target_was_home_list.append(target_row.iloc[0]['was_home'])
        except:
            target_points_list.append(np.nan)
            target_fdr_list.append(np.nan)
            target_was_home_list.append(None)

    except:
        total_points_2_list.append(np.nan)
        ict_2_list.append(np.nan)
        expected_goal_involvements_2_list.append(np.nan)
        expected_goals_conceded_2_list.append(np.nan)
        minutes_2_list.append(np.nan)

        target_points_list.append(np.nan)
        target_fdr_list.append(np.nan)
        target_was_home_list.append(None)

#Now add them to the player_data_df
players_data_df['target_points'] = target_points_list
players_data_df['target_fdr'] = target_fdr_list
players_data_df['target_was_home'] = target_was_home_list

players_data_df['total_points_2'] = total_points_2_list
players_data_df['ict_index_2'] = ict_2_list
players_data_df['expected_goal_involvements_2'] = expected_goal_involvements_2_list
players_data_df['expected_goals_conceded_2'] = expected_goals_conceded_2_list
players_data_df['minutes_2'] = minutes_2_list

#Inputs -  opponent_team, total_points, was_home, minutes, goals_scored, assists, bps, influence, creativity, threat, clearances_blocks_interceptions, recoveries, tackles, defensive_contribution, expected_goal_involvements
#Target - Next GW total_point

players_data_df = players_data_df[['element', 'total_points', 'total_points_2', 'minutes', 'minutes_2',
                                   'ict_index', 'ict_index_2', 'expected_goal_involvements',
                                   'expected_goal_involvements_2', 'expected_goals_conceded', 'expected_goals_conceded_2',
                                   'target_was_home', 'target_fdr', 'target_points']]

players_data_df = players_data_df.dropna(subset='target_points') #Drops all the rows without targets

#Merge all relevant data into one df called gameweek info
#Merge players_data and player dfs on id/element
players_data_df = players_data_df.merge(players_df,how = 'inner', left_on='element', right_on='id')
gameweek_info = players_data_df.drop(columns = ['element', 'id']) #Final df created with inputs and targets

train_df, val_df = train_test_split(gameweek_info, test_size = 0.2, random_state=42)



#Model Training
'''
Identify inputs & targets - ✓
Identify numerical & categorical columns - ✓
Impute missing values - ✓
Scale numeric values 0,1 range - ✓
Encode categorical columns - ✓
'''
train_inputs = train_df.drop('target_points', axis = 1)
train_targets = train_df['target_points'] # Set inputs and targets

val_inputs = val_df.drop('target_points', axis = 1)
val_targets = val_df['target_points']

numeric_cols = ['total_points', 'total_points_2', 'minutes', 'minutes_2', 'ict_index', 'ict_index_2', 'expected_goal_involvements', 'expected_goal_involvements_2', 'expected_goals_conceded', 'expected_goals_conceded_2']
categoric_cols =['target_was_home', 'element_type']

#Scale the numeric features
scaler = MinMaxScaler().fit(train_inputs[numeric_cols])

train_inputs[numeric_cols]= scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols]= scaler.transform(val_inputs[numeric_cols])

#Encode the categoric columns
#Map the home/away data to binary 0/1
home_away_codes = {True:1, False:0}
train_inputs['target_was_home']=train_inputs['target_was_home'].map(home_away_codes)
val_inputs['target_was_home']=val_inputs['target_was_home'].map(home_away_codes)

#Encode the positions using one hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[['element_type']])
encoded_cols = list(encoder.get_feature_names_out(['element_type']))
train_inputs[encoded_cols]= encoder.transform(train_inputs[['element_type']])
val_inputs[encoded_cols]= encoder.transform(val_inputs[['element_type']])

#Save inputs and targets dataframes as parquet
train_inputs.to_parquet('data/splits/train-inputs.parquet')
val_inputs.to_parquet('data/splits/val-inputs.parquet')
pd.DataFrame(train_targets).to_parquet('data/splits/train-targets.parquet', index = False)
pd.DataFrame(val_targets).to_parquet('data/splits/val-targets.parquet', index = False)
