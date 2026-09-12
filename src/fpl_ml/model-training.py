import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

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
og_train_inputs = pd.read_parquet('data/splits/train-inputs.parquet')
og_val_inputs = pd.read_parquet('data/splits/train-inputs.parquet')
og_train_targets = pd.Series(pd.read_parquet('data/splits/train-targets.parquet')['target_points'])
og_val_targets = pd.Series(pd.read_parquet('data/splits/train-targets.parquet')['target_points'])

#Baseline models
baseline_predictions_mean = [og_train_targets.mean()]*len(og_val_targets)
baseline_predictions_last_week = og_val_inputs['total_points']
baseline_targets = og_val_targets

save_model_stats(baseline_predictions_mean, baseline_targets, 'baseline')
save_model_stats(baseline_predictions_last_week,baseline_targets, 'baseline_last_week')