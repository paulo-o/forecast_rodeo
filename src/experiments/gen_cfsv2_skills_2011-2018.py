# Generate CFSv2 debiased skills for 2011-2018

import pickle
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Ensure that working directory is forecast_rodeo
if os.path.basename(os.getcwd()) == "experiments":
    # Navigate to forecast_rodeo
    os.chdir(os.path.join("..",".."))
if os.path.basename(os.getcwd()) != "forecast_rodeo":
    raise Exception("You must be in the forecast_rodeo folder")

# Adds 'experiments' folder to path to load experiments_util
sys.path.insert(0, 'src/experiments')
# Load general utility functions
from experiments_util import *
# Load functionality for fitting and predicting
from fit_and_predict import *
# Load functionality for evaluation
from skill import *


### GET CFSv2 SKILLS FOR 2011-2017

# Get submittion dates
forecast_dates = [datetime(y, 4, 18) + timedelta(14*i) for y in range(2011, 2018) for i in range(26)]
forecast_dates = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day) for date in forecast_dates]

# Create dataframe to store skills
skills_df = pd.DataFrame(index=range(len(forecast_dates)), columns=['start_date', 'tmp2m_34', 'tmp2m_56', 'precip_34', 'precip_56'])
skills_df['forecast_date'] = forecast_dates

# Create dict to store cfsv2 forecasts, climatology, anomalies, ground truth
output_dict = {'{}_{}'.format(variable, window):
               pd.DataFrame(columns=['lat', 'lon', 'start_date', 'cfsv2_{}'.format(variable),
                                     '{}_clim'.format(variable), '{}_anom'.format(variable),
                                     'cfsv2_{}_anom'.format(variable)])
               for variable in ['tmp2m', 'precip'] for window in ['34', '56']}

# Load climatologies and gt_anomalies for tmp2m and precip
anoms_tmp2m = get_lat_lon_date_features(anom_ids=['contest_tmp2m'],
                                        first_year=get_first_year('contest_tmp2m'))
anoms_tmp2m = anoms_tmp2m[['lat', 'lon', 'start_date', 'tmp2m_clim', 'tmp2m_anom']]
anoms_precip = get_lat_lon_date_features(anom_ids=['contest_precip'],
                                         first_year=get_first_year('contest_precip'))
anoms_precip = anoms_precip[['lat', 'lon', 'start_date', 'precip_clim', 'precip_anom']]
anoms = {'tmp2m': anoms_tmp2m, 'precip': anoms_precip}

# Add skills to skills_df
for forecast_string in skills_df['forecast_date']:
    print(forecast_string)
    forecast_date = datetime.strptime(forecast_string, '%Y%m%d')

    # Load 4 files, one for each hour of the forecast dates (e.g., 2011-04-17 and 2011-04-18, 00, 06, 12, 18h)
    file = open("data/forecast/cfsv2_2011-2018/cfs_{}00.pkl".format(forecast_string), 'rb')
    data = pickle.load(file)
    file.close()

    del data['issue_datetime']  # delete extraneous column

    # Change longitude to positive, change temperature in Kelvin to Celsius
    data['longitude'] = data['longitude'] + 360
    data['t2m_K'] = data['t2m_K'] - 273.15

    # Rename columns
    data.columns = ['forecast_date', 'cfsv2_precip', 'lat', 'lon', 'cfsv2_tmp2m', 'datetime']

    for window in ['34', '56']:
        # Subset data to correct dates depending on window
        if window == '34':
            df = data.loc[(pd.to_datetime(data['datetime']) >= (forecast_date + timedelta(days=14))) &
                          (pd.to_datetime(data['datetime']) < (forecast_date + timedelta(days=28)))]
        else:
            df = data.loc[(pd.to_datetime(data['datetime']) >= (forecast_date + timedelta(days=28))) &
                          (pd.to_datetime(data['datetime']) < (forecast_date + timedelta(days=42)))]

        # Average over given latitude and longitude
        df = df.groupby(['lat', 'lon']).mean().reset_index()

        # Use cumulative precipitation (data is 6-hourly, so x4 for daily, x14 for biweekly)
        df['cfsv2_precip'] = df['cfsv2_precip'] * 4 * 14

        # Add start date to dataframe (first day of target period)
        if window == '34':
            df['start_date'] = forecast_date + timedelta(days=14)
        else:
            df['start_date'] = forecast_date + timedelta(days=28)

        for variable in ['tmp2m', 'precip']:
            # Subset dataframe to relevant variables and use contest mask
            df_subset = df[['lat', 'lon', 'start_date', 'cfsv2_{}'.format(variable)]]
            df_subset = subsetmask(df_subset, mask_df=get_contest_mask())

            # Merge gt climatologies
            df_subset = pd.merge(df_subset, anoms[variable], on=['lat', 'lon', 'start_date'], how='left')
            # Create forecast anomalies by subtracting cfsv2 prediction from gt climatology
            df_subset['cfsv2_{}_anom'.format(variable)] = df_subset['cfsv2_{}'.format(variable)] - df_subset['{}_clim'.format(variable)]

            # Add cfsv2 forecast, gt climatology, and cfsv2 and gt anomalies to output dictionary
            output_dict['{}_{}'.format(variable, window)] = output_dict['{}_{}'.format(variable, window)].append(df_subset, ignore_index=True)

            # Calculate skill for given forecast date
            skill = get_col_skill(df_subset,
                                  gt_anomaly_col='{}_anom'.format(variable),
                                  forecast_anomaly_col='cfsv2_{}_anom'.format(variable),
                                  time_average=True)

            # Store skill in dataframe and print it
            skills_df.at[skills_df['forecast_date'] == forecast_string, '{}_{}'.format(variable, window)] = skill
            print('Skill for {}_{}: {}'.format(variable, window, skill))


### OUTPUT DATAFRAMES FOR FORECASTS AND SKILLS

for variable in ['tmp2m', 'precip']:
    for window in ['34', '56']:
        forecast_df = output_dict['{}_{}'.format(variable, window)]
        forecast_df.to_hdf('results/skills/cfsv2/cfsv2_{}_{}_forecast_2011-2018.h5'.format(variable, window), key='data', mode='w')
skills_df.to_hdf('results/skills/cfsv2/cfsv2_skills_2011-2018.h5', key='data', mode='w')


### GET DEBIASED CFSv2 SKILLS

# Pick variable and window
variable = 'precip'
window = '56'

for variable in ['precip', 'tmp2m']:
    for window in ['34', '56']:

        gt_id = 'contest_' + variable
        target_horizon = window +'w'

        # Load CFSv2 predictions
        cfsv2 = pd.read_hdf('results/skills/cfsv2/cfsv2_{}_{}_forecast_2011-2018.h5'.format(variable, window))

        # Create columns for year, month, day
        cfsv2['year'] = cfsv2['start_date'].dt.year
        cfsv2['month'] = cfsv2['start_date'].dt.month
        cfsv2['day'] = cfsv2['start_date'].dt.day


        # Get the hindcast submission and target dates
        contest_id = get_contest_id(gt_id, target_horizon)
        hindcast_template_file = os.path.join("data", "fcstrodeo_nctemplates", contest_id + "_hindcast_template.nc")
        hindcast_template = netCDF4.Dataset(hindcast_template_file)
        hindcast_submission_dates = hindcast_template['dates'][:]
        hindcast_target_dates = pd.Series([get_target_date(str(sub_date), target_horizon) for sub_date in hindcast_submission_dates])
        hindcast_submission_dates = pd.Series([datetime.strptime(str(d), "%Y%m%d") for d in hindcast_submission_dates])
        # Find all unique day-month combinations
        hindcast_day_months = pd.DataFrame({'month' : hindcast_submission_dates.dt.month,
                                            'day': hindcast_submission_dates.dt.day}
                                          ).drop_duplicates()

        # column names for gt_col, clim_col and anom_col
        measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'prate'
        gt_col = measurement_variable
        clim_col = measurement_variable+"_clim"
        anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'prate_anom'

        # Load CFSv2 reforecast, average over rows
        mod_gt_id = gt_id if gt_id == 'contest_tmp2m' else 'contest_prate'
        cfsv2r = pd.read_hdf("data/dataframes/cfsv2_re-{}-{}.h5".format(mod_gt_id, target_horizon))
        # cfsv2r = cfsv2r[cfsv2r.start_date.isin(hindcast_target_dates)]
        cfsv2r["cfsv2_re"] = np.mean(cfsv2r.iloc[:, 3:], axis = 1)
        cfsv2r = cfsv2r.drop(cfsv2r.columns[3:11], axis=1)

        # Merge in ground truth, climatology, and gt anomalies
        anoms = get_lat_lon_date_features(anom_ids = [gt_id], first_year=get_first_year(gt_id))
        cfsv2r = pd.merge(cfsv2r, anoms,
                        on=['lat', 'lon', 'start_date'], how='left')

        # Form cfsv2 anomalies
        cfsv2r["cfsv2_re_anom"] = cfsv2r["cfsv2_re"] - cfsv2r[clim_col]

        # Measure error of cfsv2
        cfsv2r['cfsv2_err'] = cfsv2r[gt_col] - cfsv2r['cfsv2_re']
        # Assign start dates groups by month-day combination
        cfsv2r['month_day'] = ['{}_{}'.format(month, day) for (month, day) in zip(cfsv2r.start_date.dt.month,cfsv2r.start_date.dt.day)]

        # Compute mean error by lat, lon, month, day
        cfsv2r['cfsv2_err_mean'] = cfsv2r.groupby([cfsv2r.lat, cfsv2r.lon, cfsv2r.month_day])['cfsv2_err'].transform('mean')


        # Create month and day columns for cfsv2r
        cfsv2r['month'] = cfsv2r['start_date'].dt.month
        cfsv2r['day'] = cfsv2r['start_date'].dt.day

        # Store average error in 1999-2010 for each lat, lon, day, month
        cfsv2_correction = cfsv2r[['month', 'day', 'lat', 'lon', 'cfsv2_err_mean']]
        cfsv2_correction = cfsv2_correction.drop_duplicates()

        # Merge cfsv2_correction into cfsv2
        cfsv2 = pd.merge(cfsv2, cfsv2_correction, on=['lat', 'lon', 'month', 'day'], how='left')


        # Create debiased CFSv2 and anomalized debiased CFSv2 columns
        cfsv2['debiased_cfsv2_{}'.format(variable)] = cfsv2['cfsv2_{}'.format(variable)] + cfsv2['cfsv2_err_mean']
        cfsv2['debiased_cfsv2_{}_anom'.format(variable)] = cfsv2['debiased_cfsv2_{}'.format(variable)] - cfsv2['{}_clim'.format(variable)]
        cfsv2.head()

        cfsv2[['lat', 'lon', 'start_date', 'debiased_cfsv2_{}'.format(variable), 'debiased_cfsv2_{}_anom'.format(variable)]].to_hdf('results/skills/cfsv2/debiased_cfsv2_{}_{}_forecast_2011-2018.h5'.format(variable, window), key='data', mode='w')

        skill = get_col_skill(cfsv2,
                              gt_anomaly_col='{}_anom'.format(variable),
                              forecast_anomaly_col='debiased_cfsv2_{}_anom'.format(variable),
                              time_average=False)

        export_df = pd.DataFrame(skill).reset_index()
        export_df.columns = ['start_date', 'skill']
        export_df.to_hdf('results/skills/debiased_cfsv2/skill-contest_{}-{}w.h5'.format(variable, window), key='data', mode='w')
