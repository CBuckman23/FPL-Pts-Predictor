import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Functions
def rmse(predictions, targets):
    return np.sqrt(np.mean(np.square(predictions-targets)))

def save_model_stats(predictions, targets, model_name):
    model_rmse = rmse(predictions, targets)
    model_r2 = r2_score(targets, predictions)
    model_mae = mean_absolute_error(targets, predictions)
    rho, p_value = return_spearman_r(predictions, targets)

    results = pd.DataFrame([{
        "model_name": model_name,
        "mae": model_mae,
        "rmse": model_rmse,
        "r2": model_r2,
        "rho": rho,
        "p_value": p_value
    }])

    results.to_csv(
        'results/model_performance.csv',
        mode = 'a',
        header = not Path('results/model_performance.csv').exists(),
        index = False
    )



def return_spearman_r(predictions, targets):
    try:
        rho, p_value = spearmanr(predictions, targets)
    except:
        rho, p_value = np.nan, np.nan
    return rho, p_value

#Load inputs and targets from parquet files
baseline_val_inputs = pd.read_parquet('data/splits/val-inputs.parquet')
baseline_train_targets = pd.read_parquet('data/splits/train-targets.parquet')
baseline_val_targets = pd.read_parquet('data/splits/val-targets.parquet')

initial_train_inputs = pd.read_parquet('data/splits/initial-model/initial_train_inputs.parquet')
initial_val_inputs = pd.read_parquet('data/splits/initial-model/initial_val_inputs.parquet')
initial_train_targets =pd.read_parquet('data/splits/train-targets.parquet')
initial_val_targets = pd.read_parquet('data/splits/val-targets.parquet')
#Baseline models
baseline_predictions_mean = [baseline_train_targets.mean()]*len(baseline_val_targets) #Baseline is just predicting the mean
baseline_predictions_last_week = baseline_val_inputs['total_points_2']
baseline_targets = baseline_val_targets

save_model_stats(baseline_predictions_mean, baseline_targets, 'baseline')
save_model_stats(baseline_predictions_last_week,baseline_targets, 'baseline_last_week')

#Train models

#Initial decision tree model


initial_dec_tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(initial_train_inputs, initial_train_targets)
initial_dec_tree_val_preds = initial_dec_tree.predict(initial_val_inputs)
save_model_stats(initial_dec_tree_val_preds, initial_val_targets, 'initial_dec_tree')
joblib.dump(initial_dec_tree, 'models/initial_dec_tree.joblib')

#Initial random forest model
initial_ran_for = RandomForestRegressor(max_depth=2, random_state=42).fit(initial_train_inputs, initial_train_targets)
initial_ran_for_val_preds = initial_ran_for.predict(initial_val_inputs)
save_model_stats(initial_ran_for_val_preds, initial_val_targets, 'initial_ran_for')
joblib.dump(initial_ran_for, 'models/initial_ran_for.joblib')

'''GK model
train_inputs_gk = initial_train_inputs.loc[initial_train_inputs['element_type']==1]
val_inputs_gk = initial_val_inputs.loc[og_val_inputs['element_type']==1]
train_inputs_gk = train_inputs_gk[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]
val_inputs_gk = val_inputs_gk[['total_points', 'total_points_2', 'minutes', 'minutes_2', 'expected_goals_conceded', 'expected_goals_conceded_2','target_fdr', 'target_was_home']]

#train_targets_gk = og_train_targets.loc[og_train_df['element_type']==1]['target_points']
#val_targets_gk = val_df.loc[val_df['element_type']==1]['target_points']'''
print(initial_train_inputs.info())