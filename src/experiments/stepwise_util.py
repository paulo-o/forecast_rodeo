# Supporting functionality for stepwise regression
import sys
# Adds 'experiments' folder to path to load experiments_util
sys.path.insert(0, 'src/experiments')
# Load general utility functions
from experiments_util import *

def default_result_file_names(gt_id = "contest_tmp2m", 
                              target_horizon = "34w", 
                              margin_in_days = 56,
                              criterion = "similar_mean",
                              submission_date_str = "19990418",
                              experiment = "regression",
                              procedure = "forward_stepwise",
                              hindcast_folder = True,
                              hindcast_features = True,
                              use_knn1 = False):
    """Returns default result file names for stepwise regression

    Args:
       gt_id: "contest_tmp2m" or "contest_precip"
       target_horizon: "34w" or "56w"
       margin_in_days
       criterion
       submission_date_str
       experiment (optional)
       procedure: "forward_stepwise" or "backward_stepwise"
       hindcast_folder: if True, subfolder is called "hindcast", else "contest_period"
       hindcast_features: if True, use hindcast features (smaller set), else use forecast features
       use_knn1: if True, add knn1 to set of candidate x cols
    """
    # Get default candidate predictors
    initial_candidate_x_cols = default_stepwise_candidate_predictors(gt_id, target_horizon, hindcast=hindcast_features)
    if use_knn1:
        initial_candidate_x_cols = initial_candidate_x_cols + ['knn1']
    # Build identifying parameter string
    param_str = 'margin{}-{}-{}'.format(
    margin_in_days, criterion, str(abs(hash(frozenset(initial_candidate_x_cols)))))
    # Create directory for storing results
    outdir = os.path.join('results',experiment,'hindcast' if hindcast_folder else 'contest_period',
                          gt_id+'_'+target_horizon,procedure,
                          param_str)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Return dictionary of result file names
    return { "path_preds" : os.path.join(outdir, submission_date_str+'.h5'),
            "path_stats" : os.path.join(outdir, 'stats-'+submission_date_str+'.pkl'),
            "converged" : os.path.join(outdir, 'converged-'+submission_date_str)}

def default_stepwise_candidate_predictors(gt_id, target_horizon, hindcast=True):
    """Returns default set of candidate predictors for stepwise regression

    Args:
       gt_id: "contest_tmp2m" or "contest_precip"
       target_horizon: "34w" or "56w"
       hindcast: if True, cannot use anom predictors because they involve climatology
    """
    # Identify measurement variable name
    measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'
    # column names for gt_col, clim_col and anom_col 
    clim_col = measurement_variable+"_clim"
    # temperature, 3-4 weeks
    if gt_id == "contest_tmp2m" and target_horizon == "34w":
        if hindcast:
            candidate_x_cols = ['ones', 'tmp2m_shift29', 'tmp2m_shift58', 
                                ###'tmp2m_shift29_anom', 'tmp2m_shift58_anom', 'tmp2m_shift365_anom', 
                                'rhum_shift30',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
                                'mei_shift45', 'phase_shift17',
                                'sst_1_shift30', 'sst_2_shift30', 'sst_3_shift30',
                                'icec_1_shift30', 'icec_2_shift30', 'icec_3_shift30',
                                'wind_hgt_10_1_shift30', 'wind_hgt_10_2_shift30']
        else:
            candidate_x_cols = ['ones', 'tmp2m_shift29', 'tmp2m_shift29_anom', 'tmp2m_shift58', 'tmp2m_shift58_anom',
                                'rhum_shift30', 'pres_shift30',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
                                'mei_shift45', 'phase_shift17',
                                # 'sst_1_shift30', 'sst_2_shift30', 'sst_3_shift30', 
                                'sst_2010_1_shift30', 'sst_2010_2_shift30', 'sst_2010_3_shift30', 
                                # 'icec_1_shift30', 'icec_2_shift30', 'icec_3_shift30', 
                                'icec_2010_1_shift30', 'icec_2010_2_shift30', 'icec_2010_3_shift30', 
                                # 'wind_hgt_10_1_shift30', 'wind_hgt_10_2_shift30',
                                'wind_hgt_10_2010_1_shift30', 'wind_hgt_10_2010_2_shift30']
    #--------------- 
    # temperature, 5-6 weeks
    if gt_id == "contest_tmp2m" and target_horizon == "56w":
        if hindcast:
            candidate_x_cols = ['ones', 'tmp2m_shift43', 'tmp2m_shift86', 
                                'tmp2m_shift365', 
                                ###'tmp2m_shift86_anom', 'tmp2m_shift43_anom', 'tmp2m_shift365_anom', 
                                'rhum_shift44', 'pres_shift44', ###'pevpr_shift44',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
                                'mei_shift59', 'phase_shift31',
                                'sst_1_shift44', 'sst_2_shift44', 'sst_3_shift44',
                                'icec_1_shift44',
                                'wind_hgt_10_1_shift44', 'wind_hgt_10_2_shift44']
            # Optionally, add anomaly PCs of pres, rhum, and slp
            ###candidate_x_cols = candidate_x_cols + ['pres_anom_1_shift44', 
            ###                    'rhum_anom_2_shift44']
        else:
            candidate_x_cols = ['ones', 'tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                                'rhum_shift44', 'pres_shift44',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
                                'mei_shift59', 'phase_shift31',
                                # 'sst_1_shift44', 'sst_2_shift44', 'sst_3_shift44', 
                                'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44', 
                                # 'icec_1_shift44', 'icec_2_shift44', 'icec_3_shift44', 
                                'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44', 
                                # 'wind_hgt_10_1_shift44', 'wind_hgt_10_2_shift44',
                                'wind_hgt_10_2010_1_shift44', 'wind_hgt_10_2010_2_shift44']
    #--------------- 
    # precipitation, 3-4 weeks
    if gt_id == "contest_precip" and target_horizon == "34w":
        if hindcast:
            candidate_x_cols = ['ones', ###'tmp2m_shift29_anom', 'tmp2m_shift58_anom',
                                'rhum_shift30', 'pres_shift30', 'pres_shift60',
                                'precip_shift29', ###'precip_shift29_anom', 'precip_shift58_anom',
                                'precip_shift58', 'mei_shift45', 'phase_shift17',
                                'sst_1_shift30', 'sst_2_shift30', 
                                'icec_2_shift30', 
                                'wind_hgt_10_1_shift30', 'wind_hgt_10_2_shift30',
                                'wind_hgt_850_1_shift30', 'wind_hgt_850_2_shift30']
            # Optionally, add anomaly PCs of pres, rhum, and slp
            ###candidate_x_cols = candidate_x_cols + ['pres_anom_1_shift30', 'pres_anom_2_shift30', 
            ###                    'rhum_anom_1_shift30', 'rhum_anom_2_shift30']
        else:
            candidate_x_cols = ['ones', 'tmp2m_shift29', 'tmp2m_shift29_anom', 'tmp2m_shift58', 'tmp2m_shift58_anom',
                                'rhum_shift30', 'pres_shift30',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
                                'precip_shift29', 'precip_shift29_anom', 'precip_shift58', 'precip_shift58_anom',
                                'mei_shift45', 'phase_shift17',
                                # 'sst_1_shift30', 'sst_2_shift30', 'sst_3_shift30', 
                                'sst_2010_1_shift30', 'sst_2010_2_shift30', 'sst_2010_3_shift30', 
                                # 'icec_1_shift30', 'icec_2_shift30', 'icec_3_shift30', 
                                'icec_2010_1_shift30', 'icec_2010_2_shift30', 'icec_2010_3_shift30', 
                                # 'wind_hgt_10_1_shift30', 'wind_hgt_10_2_shift30',
                                'wind_hgt_10_2010_1_shift30', 'wind_hgt_10_2010_2_shift30']
    #--------------- 
    # precipitation, 5-6 weeks
    if gt_id == "contest_precip" and target_horizon == "56w":
        if hindcast:
            candidate_x_cols = ['ones', 'tmp2m_shift43', 'tmp2m_shift86', 
                                ###'tmp2m_shift43_anom', 'tmp2m_shift86_anom',
                                'rhum_shift44', 'pres_shift44',
                                'nmme0_wo_ccsm3_nasa', 
                                'precip_shift43', 'precip_shift86', 
                                ###'precip_shift43_anom', 'precip_shift86_anom',
                                'mei_shift59', 'phase_shift31',
                                'wind_hgt_10_1_shift44']
            # Optionally, add anomaly PCs of pres, rhum, and slp
            ###candidate_x_cols = candidate_x_cols + ['pres_anom_1_shift44', 'pres_anom_2_shift44',
            ###                    'rhum_anom_1_shift44', 'rhum_anom_2_shift44',
            ###                    'slp_anom_1_shift44', 'slp_anom_2_shift44']
        else:
            candidate_x_cols = ['ones', 'tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                                'rhum_shift44', 'pres_shift44',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
                                'precip_shift43', 'precip_shift43_anom', 'precip_shift86', 'precip_shift86_anom',
                                'mei_shift59', 'phase_shift31',
                                # 'sst_1_shift44', 'sst_2_shift44', 'sst_3_shift44', 
                                'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44', 
                                # 'icec_1_shift44', 'icec_2_shift44', 'icec_3_shift44', 
                                'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44', 
                                # 'wind_hgt_10_1_shift44', 'wind_hgt_10_2_shift44',
                                'wind_hgt_10_2010_1_shift44', 'wind_hgt_10_2010_2_shift44']
    return candidate_x_cols