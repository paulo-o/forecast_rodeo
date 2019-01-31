# Supporting functionality for knn
import sys
from datetime import datetime
# Adds 'experiments' folder to path to load experiments_util
sys.path.insert(0, 'src/experiments')
# Load general utility functions
from experiments_util import *

def get_last_holdout_date(target_date_obj, target_horizon, rolling_hindcast=False):
    """Returns the last date (inclusive) of the hold-out period associated 
    with the given target date
    
    Args:
        target_date_obj: start date of the target forecasting period
        target_horizon: '34w' or '56w'
        rolling_hindcast: if False, last hold-out date is given by the April 17th following
           the submission date associated with target_date_obj; 
           otherwise, last hold-out date is computed by identifying the official submission date 
           associated with target_date_obj, setting the year to the following year, and subtracting one day
    """
    # Compute associated submission date
    submission_date = target_date_obj - timedelta(get_deadline_delta(target_horizon))
    if not rolling_hindcast:
        # Hold-out year defined in terms of submission dates and ends on April 17
        last_holdout_date = datetime(month=4, day=17, year=submission_date.year)
        return (last_holdout_date if submission_date <= last_holdout_date 
                else last_holdout_date.replace(year=submission_date.year+1))
    else:
        if not ((submission_date.month == 2) and (submission_date.day == 29)):            
            # Map submission date to following year and subtract one day
            return submission_date.replace(year=submission_date.year+1) - timedelta(1)
        else:
            # If submission date is a leap day, subtract one day and then map to following year
            return (submission_date - timedelta(1)).replace(year=submission_date.year+1) 
            
def get_target_neighbors(target_date_obj, target_horizon, gt_id, 
                         nbr_start_delta, past_days, viable_similarities, 
                         hindcast_mode=True, rolling_hindcast=False):
    """Returns the viable neighbors of a target date ordered by decreasing similarity
    
    Args:
        target_date_obj: start date of the target forecasting period
        target_horizon: '34w' or '56w'
        gt_id: ground truth data string ending in "precip" or "tmp2m"
        nbr_start_delta: minimum number of days between start date of most recent neighbor to consider
            and start date of target period 
        past_days: the number of past days that should contribute to measure of similarity
        viable_similarities: similarities of neighbors with available ground truth data
        hindcast_mode: run in fixed year hindcast mode? if True, viable neighbors defined by
           contest hindcast hold-out rules; otherwise, viable neighbors defined by
           those fully observable on the submission date associated with target date
           when rolling_hindcast is False and defined by rolling hindcast window when
           rolling hindcast is True
        rolling_hindcast: see get_last_holdout_date
    """
    # Identify the similarities to this target date
    target_sims = viable_similarities[target_date_obj]
    # Consider a neighbor viable if
    # (1) the neighbor ground truth measurement is fully observable on the submission 
    # date associated with target date, i.e., neighbor date <= last observable start date
    # on the associated submission date
    last_observable_start_date = target_date_obj - timedelta(get_start_delta(target_horizon, gt_id))
    viable = target_sims.index <= last_observable_start_date
    if hindcast_mode or rolling_hindcast:
        # Compute the last holdout period for this target_date
        last_holdout_date = get_last_holdout_date(target_date_obj, target_horizon, rolling_hindcast)
        # OR (2) the measurements contributing to neighbor's similarity all occur after last_hold_out_date
        viable = viable | (
            target_sims.index > (last_holdout_date + timedelta(nbr_start_delta + past_days - 1)))
        # OR (3) the neighbor ground truth measurement occurs after last_hold_out_date
        # and the latest measurement contributing to neighbor similarity is fully observable on the 
        # submission date (i.e., start date of latest neighbor similarity measurement <= 
        # last observable start date on the associated submission date
        viable = viable | ((target_sims.index > last_holdout_date) &
            (target_sims.index  <= (last_observable_start_date + timedelta(nbr_start_delta)))
        )
    # Order the viable neighbor start dates by decreasing similarity 
    return target_sims[target_sims.notnull() & viable].sort_values(ascending=False).index