## Improving Subseasonal Forecasting in the Western US with Machine Learning

Code for reproducing the results in Hwang et al. [Improving Subseasonal Forecasting in the Western US with Machine Learning](https://arxiv.org/abs/1809.07394).  Please execute all instructions, scripts, and notebooks from the base directory of the repository, i.e., the directory in which README.md is located.

### Environment and packages

The code was tested using Python 2.7 on Linux and macOS, and Anaconda 2.3.0. The following should be run to install the necessary packages:

`conda install --channel https://conda.anaconda.org/conda-forge pygrib`

`conda install netCDF4`

`conda install jpeg`

`conda install pandas`

`conda install jupyter`

`conda install scipy`

`pip install https://github.com/jcrudy/py-earth/archive/master.zip`

`conda install -c r r`

`conda install -c conda-forge cdo`

`conda install -c conda-forge hdf5=1.8.18`

`conda install -c conda-forge pytables`

### Getting started

After cloning the repository, please execute the following steps in preparation for generating forecasts.

1. The folder **data/fcstrodeo_nctemplates** contains the template files provided by NOAA and the contest organizers to generate the forecasts. Within **data**, create two additional subfolders **data/dataframes** and **data/forecast/cfsv2_2011-2018**.
2. Download the SubseasonalRodeo dataset from https://doi.org/10.7910/DVN/IHBANG and place it in **data/dataframes**.
3. Download the Reconstructed Precipitation and Temperature CFSv2 Forecasts for 2011-2018 from https://doi.org/10.7910/DVN/CEFZLV. Place the files **cfsv2_re-contest_tmp2m-56w.h5**, **cfsv2_re-contest_tmp2m-34w.h5**, **cfsv2_re-contest_prate-56w.h5** and **cfsv2_re-contest_prate-34w.h5** in **data/dataframes**. Place the all the other files in **data/forecast/cfsv2_2011-2018**.
4. For each of the four forecasting tasks with ground-truth identifier in {“contest\_tmp2m”, “contest\_precip”} and target horizon in {“34w”, “56w”}, create the feature and target data matrices used by several of our methods by executing the Jupyter notebook **create\_data\_matrices.ipynb** with `gt_id` set to equal to the ground-truth identifier and `target_horizon` set equal to the target horizon.

### Generating the MultiLLR (local linear regression with multitask model selection) forecasts

To generate MultiLLR forecasts for a ground-truth identifier in {“contest\_tmp2m”, “contest\_precip”}, a target horizon in {“34w”, “56w”}, and all target dates, execute the Jupyter notebook **batch\_2011-2018\_backward\_stepwise.ipynb** with `gt_id` set to equal to the ground-truth identifier and `target_horizon` set equal to the target horizon. This notebook, for each target date in 2011-2018, generates MultiLLR forecasts for the target date using the script **2011-2018\_backward\_stepwise.py**. Since each target date job is long-running, we recommend submitting these jobs to a cluster by setting `run_locally` to `False` and setting `batch_script` to your personal batch cluster submission script. Alternatively, you can run the jobs locally and sequentially by setting `run_locally` to `True` (in which case the setting of `batch_script` is irrelevant).

### Generating the AutoKNN (multitask k-nearest neighbor autoregression) forecasts

To generate the AutoKNN forecasts for a ground-truth identifier in {“contest\_tmp2m”, “contest\_precip”} and a target horizon in {“34w”, “56w”}, 

1. Execute the Jupyter notebook **knn\_step\_1-compute_similarities.ipynb** with `gt_id` set equal to the ground-truth identifier and `target_horizon` set equal to the target horizon.  This will compute and save the similarities between every pair of dates in the dataset.
2. Execute the Jupyter notebook **knn\_step\_2-get_neighbor\_predictions.ipynb** with `gt_id` set equal to the ground-truth identifier and `target_horizon` set equal to the target horizon. This will compute and save the predictions of the most similar viable neighbors of each target date in the dataset. 
3. Execute the Jupyter notebook **2011-2018\_regression.ipynb** with `gt_id` set equal to the ground-truth identifier and `target_horizon` set equal to the target horizon. This will carry out the AutoKNN weighted local least-squares regression onto the top nearest neighbor predictions, an intercept, and fixed lagged measurements and save forecasts for all 2011-2018 target dates.

### Generating the reconstructed debiased CFSv2 forecasts

To recreate the debiased CFSv2 skills for 2011-2018, run **gen\_cfsv2\_skills\_2011-2018.py**.

### Ensembling the forecasts

To generate ensemble forecasts based on the predictions of MultiLLR, AutoKNN, and reconstructed debiased CFSv2, for a ground-truth identifier in {“contest\_tmp2m”, “contest\_precip”} and a target horizon in {“34w”, “56w”}, execute the Jupyter notebook **ensemble\_backward\_stepwise\_and\_knn\_regression.ipynb** with `gt_id` set equal to the ground-truth identifier and `target_horizon` set equal to the target horizon.

### Generating the skill tables

After completing all of the previous steps, executing the scripts **table\_skills\_contest\_year\_all\_methods.ipynb** and **table\_skills\_by\_year\_all\_methods.ipynb** will generate LaTeX tables corresponding to Tables 1 and 2 in the paper.

### Auxiliary files

- **experiments\_util.py:** utility functions supporting experiments.
- **fit\_and\_predict.py:** functionality for fitting models and forming predictions.
- **knn\_util.py:** supporting functionality for knn notebooks.
- **skill.py:** supporting functionality for evaluating predictions.
- **stepwise\_util.py:** supporting functionality for stepwise regression.

