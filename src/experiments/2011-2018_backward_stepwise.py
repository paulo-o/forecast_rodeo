# 2011-2018_backward_stepwise.py
# Generate MultiLLR (local linear regression with multitask model selection) forecasts for a single date
# in 2011-2018.
#
# Example usage: 
#   python src/experiments/2011-2018_backward_stepwise.py contest_tmp2m 34w 56 mean False 20170418 16
#
# Args:
#   gt_id: string identifier for which data to forecast ("contest_precip" or "contest_tmp2m")
#   target_horizon: "34w" or "56w"
#   margin_in_days: regression will train on all dates within margin_in_days of the target day and month
#     in each training year; set to 0 if you only want to train on the target day and month in each year
#   criterion: string criterion to use for backward stepwise (use "mean" to reproduce results in paper). 
#     Choose from: mean, mean_over_sd, similar_mean, similar_mean_over_sd, similar_quantile_0.5,
#     similar_quantile_0.25, similar_quantile_0.1 or create your own in the function 
#     skill_report_summary_stats in skill.py
#   hindcast_features: True to use smaller set of hindcast features, False to use full set of features
#     (use False to reproduce results in paper)
#   submission_date_str: string submission date for prediction in "YYYYMMDD" format
#   num_cores: number of cores to use in execution

import sys
import numpy as np
import pandas as pd
from sklearn import *
import subprocess
from datetime import datetime, timedelta
import netCDF4
import time
from functools import partial
import os
import gc
import pickle
# Adds 'experiments' folder to path to load experiments_util
sys.path.insert(0, 'src/experiments')
# Load general utility functions
from experiments_util import *
# Load functionality for fitting and predicting
from fit_and_predict import *
# Load functionality for evaluation
from skill import *
# Load functionality for stepwise regression
from stepwise_util import *

# Command-line inputs
gt_id = "contest_tmp2m" if len(sys.argv) < 2 else sys.argv[1]
print gt_id
target_horizon = "34w" if len(sys.argv) < 3 else sys.argv[2]
print target_horizon
margin_in_days = 56 if len(sys.argv) < 4 else int(sys.argv[3])
print margin_in_days
criterion = 'mean' if len(sys.argv) < 5 else sys.argv[4]
print criterion
hindcast_features = False if len(sys.argv) < 6 else (sys.argv[5] == "True")
print hindcast_features
submission_date_str = '20110418' if len(sys.argv) < 7 else sys.argv[6]
print submission_date_str
submission_date_obj = datetime.strptime(submission_date_str, "%Y%m%d")
# Get target date as a datetime object
target_date_obj = get_target_date(submission_date_str, target_horizon)
print 'target date: {}'.format(target_date_obj)
num_cores = 16 if len(sys.argv) < 8 else int(sys.argv[7])
print num_cores

if hindcast_features:
    print "Using hindcast features"
else:
    print "Using forecast features"
    
# Experiment name
experiment = "regression"

# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

# column names for gt_col, clim_col and anom_col 
gt_col = measurement_variable
clim_col = measurement_variable+"_clim"
anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'precip_anom'

# anom_inv_std_col: column name of inverse standard deviation of anomalies for each start_date
anom_inv_std_col = anom_col+"_inv_std"

#
# Default regression parameter values
#
# anom_scale_col: multiply anom_col by this amount prior to prediction
# (e.g., 'ones' or anom_inv_std_col)
anom_scale_col = 'ones'
# pred_anom_scale_col: multiply predicted anomalies by this amount
# (e.g., 'ones' or anom_inv_std_col)
pred_anom_scale_col = 'ones'
# choose first year to use in training set
first_train_year = 1948 if gt_id == 'contest_precip' else 1979
# columns to group by when fitting regressions (a separate regression
# is fit for each group); use ['ones'] to fit a single regression to all points
group_by_cols = ['lat', 'lon']
# base_col: column which should be subtracted from gt_col prior to prediction
# (e.g., this might be clim_col or a baseline predictor like NMME)
base_col = 'zeros'
#
# Default stepwise parameter values
#
# Define candidate predictors
initial_candidate_x_cols = default_stepwise_candidate_predictors(gt_id, target_horizon, hindcast=hindcast_features)
# Copy the list of candidates for later modification
candidate_x_cols = initial_candidate_x_cols[:]
# Skill threshold for what counts as a similar year
similar_years_threshold = 0.1
# Tolerance for convergence: if improvement is less than tolerance, terminate.
tolerance = 0.01
# Whether to use margin days (days around the target date)
use_margin = False


#
# Define a string identifier for experiment parameters
#
param_str = 'margin{}-{}-{}'.format(
    margin_in_days, criterion, str(abs(hash(frozenset(initial_candidate_x_cols)))))
# Create directory for storing results
outdir = os.path.join('results',experiment,'2011-2018',
                      gt_id+'_'+target_horizon,'backward_stepwise',
                      param_str)
if not os.path.exists(outdir):
    os.makedirs(outdir)
# When algorithm has converged create file indicating convergence
converged_outfile = os.path.join(outdir, 'converged-'+submission_date_str)
if os.path.exists(converged_outfile):
    # If algorithm has already converged previously, exit
    print '{} already exists; exiting.'.format(converged_outfile)
    sys.exit()

#
# Define similar years
#
tic()
similar_years = get_similar_years_hindcast(
    gt_col, target_horizon, target_date = target_date_obj,
    margin_days=60, experiment="regression", regen=False, hindcast=False)
toc()

#
# Create dataset
#
# Form predictions for each grid point using rolling and hindcast linear regression
prediction_func = rolling_linear_regression_wrapper
# Keep track of names of prediction columns
pred_cols = ['hindcast','forecast']
# Get number of days between start date of observation period used for prediction
# (2 weeks ahead) and start date of target period (2 or 4 weeks ahead)
start_delta = get_start_delta(target_horizon) # 29 or 43
# Compute the last date that can be used for training
last_train_date = target_date_obj - timedelta(start_delta)

# Load data superset from file
tic()
data_dir = os.path.join("results", experiment, "shared", gt_id + "_" + target_horizon)
date_data = pd.read_hdf(os.path.join(data_dir, "date_data-" + gt_id + "_" + target_horizon + ".h5"))
lat_lon_date_data = pd.read_hdf(os.path.join(data_dir, "lat_lon_date_data-" + gt_id + "_" + target_horizon + ".h5"))
toc()
# Restrict data to relevant columns and years >= first_train_year
tic()
relevant_cols = set(candidate_x_cols+[base_col,clim_col,anom_col,'start_date','lat','lon']+group_by_cols)
data = lat_lon_date_data.loc[lat_lon_date_data.start_date.dt.year >= first_train_year,
                             lat_lon_date_data.columns.isin(relevant_cols)]
data = pd.merge(data, date_data.loc[date_data.start_date.dt.year >= first_train_year,
                                    date_data.columns.isin(relevant_cols)],
                on="start_date", how="left")
del lat_lon_date_data
del date_data
toc()
# Restrict data to margin around target date and add a few additional features
tic()
sub_data = month_day_subset(data, target_date_obj, margin_in_days).copy()
del data
# Remove data past the target date so we don't train on it
sub_data = sub_data.loc[sub_data.start_date <= target_date_obj,:]
sub_data['year'] = sub_data.start_date.dt.year
sub_data['ones'] = 1.0
sub_data['zeros'] = 0.0
# To minimize the mean-squared error between predictions of the form
# (f(x_cols) + base_col - clim_col) * pred_anom_scale_col
# and a target of the form anom_col * anom_scale_col, we will
# estimate f using weighted least squares with datapoint weights
# pred_anom_scale_col^2 and effective target variable 
# anom_col * anom_scale_col / pred_anom_scale_col + clim_col - base_col
sub_data['sample_weight'] = sub_data[pred_anom_scale_col]**2
# Ensure that we do not divide by zero when dividing by pred_anom_scale_col
sub_data['target'] = (sub_data[clim_col] - sub_data[base_col] + 
                      sub_data[anom_col] * sub_data[anom_scale_col] / 
                      (sub_data[pred_anom_scale_col]+(sub_data[pred_anom_scale_col]==0)))
# Subset to datapoints with valid target and sample weights
sub_data = sub_data.dropna(subset=['target','sample_weight'])
toc()
print sub_data.head()
# Print warning if not all x columns were included
s = [x for x in candidate_x_cols if x not in sub_data.columns.tolist()]
if s:
    print "These x columns were not found:"
    print s

#
# Fit backward stepwise regression
#
# Store target-date predictions and summary stats of each model on stepwise path 
preds_outfile = os.path.join(outdir, submission_date_str+'.h5')
stats_outfile = os.path.join(outdir, 'stats-'+submission_date_str+'.pkl')
if os.path.exists(preds_outfile) and os.path.exists(stats_outfile):
    # If preds and stats already exist on disk, load them and start from the latest model
    print 'Loading existing predictions and stats'
    path_preds = pd.read_hdf(preds_outfile, key="data")
    path_stats = pickle.load( open( stats_outfile, "rb" ) )
    # The set of predictors in the latest model is specified by the last column name in
    # path_preds
    current_x_col_str = path_preds.columns[-1]
    current_x_col_set = eval(current_x_col_str)
    x_cols_current = list(current_x_col_set)
    print x_cols_current
    # Enumerate non-model columns in path_preds
    non_model_cols = ['lat','lon','start_date','truth','clim']
    # Find the last feature removed from x cols
    best_x_col = set(candidate_x_cols).symmetric_difference(current_x_col_set).pop() if (len(path_preds.columns)-len(non_model_cols)) == 1 else current_x_col_set.symmetric_difference(eval(path_preds.columns[-2])).pop()
    print best_x_col
    # Reconstruct the current best criterion
    best_criterion_current = path_stats[current_x_col_str][criterion][best_x_col]
    print best_criterion_current
else:
    path_preds = sub_data.loc[sub_data.start_date == target_date_obj, 
                              ['lat','lon','start_date',anom_col,clim_col]].copy()
    path_preds = path_preds.rename(index=str, columns={anom_col: "truth", clim_col: "clim"})
    path_stats = {}
    x_cols_current = candidate_x_cols[:]
    best_criterion_current = -np.inf

converged = False

while not converged:
    tic()
    gc.collect()
    toc()
    criteria = {}
    if not x_cols_current:
        converged = True
        break
        
    relevant_cols = set(
        x_cols_current+[base_col,clim_col,anom_col,'sample_weight','target',
                        'start_date','lat','lon','year','ones']+group_by_cols)
    print "Fitting model with core predictors {}".format(x_cols_current); 
    tic()
    preds = apply_parallel(
        sub_data.loc[:,relevant_cols].groupby(group_by_cols),
        backward_rolling_linear_regression_wrapper, num_cores,
        x_cols=x_cols_current, 
        base_col=base_col, 
        clim_col=clim_col, 
        anom_col=anom_col, 
        last_train_date=last_train_date,
        return_pseudotruth=False)
    toc()
    preds = preds.reset_index()
    tic()
    skills = skill_report(preds, target_date_obj, 
                          pred_cols=x_cols_current, 
                          gt_anom_col='truth',
                          clim_col='clim',
                          include_trunc0 = gt_id.endswith("precip"),
                          include_cos_margin = use_margin, 
                          verbose = False)
    # Remove the target year from the skills dataframe so it isn't used in evaluation
    skills['cos'] = skills['cos'][skills['cos'].index != target_date_obj]
    toc()
    summary_stats = skill_report_summary_stats_multicol(
        skills, similar_years, threshold=similar_years_threshold, 
        use_margin=use_margin)
    # Pick best column based on summary stats
    criteria = summary_stats[criterion]
    best_x_col = criteria.argmax()
    # Compute difference from current best performance
    perf_diff = criteria[best_x_col] - best_criterion_current
    
    if perf_diff <= -tolerance:
        # Removing predictor is too costly; we're done
        converged = True
        break
        
    # Otherwise, performance hit is within tolerance:
    # Remove from model
    x_cols_current.remove(best_x_col)
    best_criterion_current = criteria[best_x_col]
    print "Removed {} from model, current criterion is {}".format(best_x_col, best_criterion_current)
    tic()
    # Store the predictions of the selected model
    path_preds = pd.merge(path_preds, 
                          preds.loc[preds.start_date == target_date_obj, 
                                    ['lat','lon',best_x_col]],
                          on=["lat","lon"], how="left"); 
    # Rename added column to reflect the set of predictors in the model
    model_str = str(set(x_cols_current))
    path_preds = path_preds.rename(
        index=str, columns = { best_x_col : model_str })
    # Store summary stats of the selected model
    path_stats[model_str] = summary_stats
    toc()
    #
    # Save predictions and summary stats to disk after each round
    #
    # Write path predictions to file
    tic()
    path_preds.to_hdf(preds_outfile, key="data", mode="w")
    toc()
    # Write path stats to file
    tic()
    f = open(stats_outfile,"wb")
    pickle.dump(path_stats,f)
    f.close()
    toc()     

    
# Mark convergence by creating converged file
print 'Saving ' + converged_outfile
open(converged_outfile, 'w').close()
