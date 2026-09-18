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
baseline_train_targets = pd.Series(pd.read_parquet('data/splits/train-targets.parquet')['target_points'])
baseline_val_targets = pd.Series(pd.read_parquet('data/splits/val-targets.parquet')['target_points'])

initial_train_inputs = pd.read_parquet('data/splits/initial-model/initial_train_inputs.parquet')
initial_val_inputs = pd.read_parquet('data/splits/initial-model/initial_val_inputs.parquet')
initial_train_targets = pd.Series(pd.read_parquet('data/splits/train-targets.parquet')['target_points'])
initial_val_targets = pd.Series(pd.read_parquet('data/splits/val-targets.parquet')['target_points'])

gk_train_inputs = pd.read_parquet('data/splits/position-models/gk_train_inputs.parquet')
gk_train_targets = pd.Series(pd.read_parquet('data/splits/position-models/gk_train_targets.parquet')['target_points'])
gk_val_inputs = pd.read_parquet('data/splits/position-models/gk_val_inputs.parquet')
gk_val_targets = pd.Series(pd.read_parquet('data/splits/position-models/gk_val_targets.parquet')['target_points'])

def_train_inputs = pd.read_parquet('data/splits/position-models/def_train_inputs.parquet')
def_train_targets = pd.Series(pd.read_parquet('data/splits/position-models/def_train_targets.parquet')['target_points'])
def_val_inputs = pd.read_parquet('data/splits/position-models/def_val_inputs.parquet')
def_val_targets = pd.Series(pd.read_parquet('data/splits/position-models/def_val_targets.parquet')['target_points'])

mid_train_inputs = pd.read_parquet('data/splits/position-models/mid_train_inputs.parquet')
mid_train_targets = pd.Series(pd.read_parquet('data/splits/position-models/mid_train_targets.parquet')['target_points'])
mid_val_inputs = pd.read_parquet('data/splits/position-models/mid_val_inputs.parquet')
mid_val_targets = pd.Series(pd.read_parquet('data/splits/position-models/mid_val_targets.parquet')['target_points'])

fwd_train_inputs = pd.read_parquet('data/splits/position-models/fwd_train_inputs.parquet')
fwd_train_targets = pd.Series(pd.read_parquet('data/splits/position-models/fwd_train_targets.parquet')['target_points'])
fwd_val_inputs = pd.read_parquet('data/splits/position-models/fwd_val_inputs.parquet')
fwd_val_targets = pd.Series(pd.read_parquet('data/splits/position-models/fwd_val_targets.parquet')['target_points'])

#Baseline models
baseline_predictions_mean = [baseline_train_targets.mean()]*len(baseline_val_targets) #Baseline is just predicting the mean
baseline_predictions_last_week = baseline_val_inputs['total_points_2']
baseline_targets = baseline_val_targets

save_model_stats(baseline_predictions_mean, baseline_targets, 'baseline_mean')
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

#Position models
gk_ran_for = RandomForestRegressor(max_depth=2, random_state=42).fit(gk_train_inputs, gk_train_targets)
gk_ran_for_val_preds = gk_ran_for.predict(gk_val_inputs)
save_model_stats(gk_ran_for_val_preds, gk_val_targets, 'gk_ran_for')

def_ran_for = RandomForestRegressor(max_depth=2, random_state=42).fit(def_train_inputs, def_train_targets)
def_ran_for_val_preds = def_ran_for.predict(def_val_inputs)
save_model_stats(def_ran_for_val_preds, def_val_targets, 'def_ran_for')

mid_ran_for = RandomForestRegressor(max_depth=2, random_state=42).fit(mid_train_inputs, mid_train_targets)
mid_ran_for_val_preds = mid_ran_for.predict(mid_val_inputs)
save_model_stats(mid_ran_for_val_preds, mid_val_targets, 'mid_ran_for')

fwd_ran_for = RandomForestRegressor(max_depth=2, random_state=42).fit(fwd_train_inputs, fwd_train_targets)
fwd_ran_for_val_preds = fwd_ran_for.predict(fwd_val_inputs)
save_model_stats(fwd_ran_for_val_preds, fwd_val_targets, 'fwd_ran_for')