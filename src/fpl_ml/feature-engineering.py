import pandas as pd
import numpy as np

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

#GK features
gk_train_inputs = train_inputs.loc[train_inputs['element_type']==1]
gk_val_inputs = val_inputs.loc[val_inputs['element_type']==1]
gk_train_inputs = gk_train_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]
gk_val_inputs = gk_val_inputs[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]

gk_train_targets = train_targets.loc[train_targets['element_type']==1]