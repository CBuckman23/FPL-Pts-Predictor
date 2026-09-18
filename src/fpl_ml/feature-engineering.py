import pandas as pd
import numpy as np

train_df = pd.read_parquet('data/raw/train-df.parquet')
val_df = pd.read_parquet('data/raw/val-df.parquet')
train_inputs = pd.read_parquet('data/splits/train-inputs.parquet')
val_inputs = pd.read_parquet('data/splits/train-inputs.parquet')
train_targets = pd.Series(pd.read_parquet('data/splits/train-targets.parquet')['target_points'])
val_targets = pd.Series(pd.read_parquet('data/splits/train-targets.parquet')['target_points'])

#Setup features
initial_train_inputs = train_inputs.drop('element_type', axis=1)
initial_val_inputs = val_inputs.drop('element_type', axis=1)
initial_train_inputs.to_parquet('data/splits/initial-model/initial_train_inputs.parquet', index = False)
initial_val_inputs.to_parquet('data/splits/initial-model/initial_val_inputs.parquet', index = False)
#targets are the entire target data frame so no change needed

#GK features & targets
gk_train_inputs = train_inputs.loc[train_inputs['element_type']==1]
gk_val_inputs = val_inputs.loc[val_inputs['element_type']==1]

gk_train_inputs = gk_train_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]
gk_val_inputs = gk_val_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]

gk_train_targets = train_df.loc[train_df['element_type']==1]['target_points']
gk_val_targets = val_df.loc[val_df['element_type']==1]['target_points']

gk_train_inputs.to_parquet('data/splits/position-models/gk_train_inputs.parquet', index = False)
gk_val_inputs.to_parquet('data/splits/position-models/gk_val_inputs.parquet', index = False)
pd.DataFrame(gk_train_targets).to_parquet('data/splits/position-models/gk_train_targets.parquet', index = False)
pd.DataFrame(gk_val_targets).to_parquet('data/splits/position-models/gk_val_targets.parquet', index = False)

#DEF features and targets
def_train_inputs = train_inputs.loc[train_inputs['element_type']==2]
def_val_inputs = val_inputs.loc[val_inputs['element_type']==2]

def_train_inputs = def_train_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]
def_val_inputs = def_val_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]

def_train_targets = train_df.loc[train_df['element_type']==2]['target_points']
def_val_targets = val_df.loc[val_df['element_type']==2]['target_points']

def_train_inputs.to_parquet('data/splits/position-models/def_train_inputs.parquet', index = False)
def_val_inputs.to_parquet('data/splits/position-models/def_val_inputs.parquet', index = False)
pd.DataFrame(def_train_targets).to_parquet('data/splits/position-models/def_train_targets.parquet', index = False)
pd.DataFrame(def_val_targets).to_parquet('data/splits/position-models/def_val_targets.parquet', index = False)

#MID features and targets
mid_train_inputs = train_inputs.loc[train_inputs['element_type']==3]
mid_val_inputs = val_inputs.loc[val_inputs['element_type']==3]

mid_train_inputs = mid_train_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goal_involvements', 'expected_goal_involvements_2',
     'target_fdr', 'target_was_home']]
mid_val_inputs = mid_val_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goal_involvements', 'expected_goal_involvements_2',
     'target_fdr', 'target_was_home']]

mid_train_targets = train_df.loc[train_df['element_type']==3]['target_points']
mid_val_targets = val_df.loc[val_df['element_type']==3]['target_points']

mid_train_inputs.to_parquet('data/splits/position-models/mid_train_inputs.parquet', index = False)
mid_val_inputs.to_parquet('data/splits/position-models/mid_val_inputs.parquet', index = False)
pd.DataFrame(mid_train_targets).to_parquet('data/splits/position-models/mid_train_targets.parquet', index = False)
pd.DataFrame(mid_val_targets).to_parquet('data/splits/position-models/mid_val_targets.parquet', index = False)

#FWD features and targets
fwd_train_inputs = train_inputs.loc[train_inputs['element_type']==4]
fwd_val_inputs = val_inputs.loc[val_inputs['element_type']==4]

fwd_train_inputs = fwd_train_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goal_involvements', 'expected_goal_involvements_2',
     'target_fdr', 'target_was_home']]
fwd_val_inputs = fwd_val_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goal_involvements', 'expected_goal_involvements_2',
     'target_fdr', 'target_was_home']]

fwd_train_targets = train_df.loc[train_df['element_type']==4]['target_points']
fwd_val_targets = val_df.loc[val_df['element_type']==4]['target_points']

fwd_train_inputs.to_parquet('data/splits/position-models/fwd_train_inputs.parquet', index = False)
fwd_val_inputs.to_parquet('data/splits/position-models/fwd_val_inputs.parquet', index = False)
pd.DataFrame(fwd_train_targets).to_parquet('data/splits/position-models/fwd_train_targets.parquet', index = False)
pd.DataFrame(fwd_val_targets).to_parquet('data/splits/position-models/fwd_val_targets.parquet', index = False)