###
# THIS FILE INCLUDES ALL THE FUNCTIONS REQUIRED TO:
# 1. PROCESS THE INPUTS OF NEURAL NETWORKS (NORMALIZE OR STANDARDIZE) IF REQUIRED
# 2. CALCULATE DIFFERENT ERFORMANCE METRICS AS PEARSON'S CORRELATION, NASHSUTCLIFF EFFICIENCY, BIAS,
#    ROOTMEAN SQUARE ERROR
# 3. ADD SYNTHETIC NOISE TO THE DATASET
# 4. CREATE DICTIONARY FOR SOME DATA STATISTICS
###

import numpy as np
from scipy.stats import pearsonr
def Data_normalize(l):
    # Description: A function to normalize array l using its mean and standard deviation
    return (l - l.mean())/l.std()

def Data_denormalize(lori, lnorm):
    # Description: A function to denormalize array lnorm using the mean and standard deviation of lori

    # Inputs:
    # lori  : The original array before normalization
    # lnorm : The normalized array
    return (lnorm * lori.std()) + lori.mean()

def Data_standize(l):
    # Description: A function to standardize array l

    return (l - l.min())/(l.max() - l.min())

def Data_destandize(lori, lstand):
    # Description: A function to de-standardize  array lstand using lori

    # Inputs:
    # lori  : The original array before standardization
    # lstand: The standardized array
    return lstand * (lori.max() - lori.min()) + lori.min()

def normalize_stat(l, mean , std):
    # Description: A function to normalize array l using mean and standard deviation defined by the user

    return (l-mean)/std

def denormalize_stat(l, mean , std):
    # Description: A function to denormalize array l using mean and standard deviation defined by the user

    return l * std + mean

def nse(predictions, targets):
    # Description: A function to compute the Nash Sutcliffe Efficiency (NSE) using predictions and targets
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))

def create_stat_dict(col_lst, df):
    # Description: A function to create a dictionary with the mean of all variables in a column list. This dictionary
    # will be later used in normalizing some continuous variables

    # Inputs:
    # col_lst:list with subset of columns in the dataframe to be included in the dictionary
    # df     : The dataframe dataset

    # Outputs:
    # stat_dict: Dictionary including the statistics(mainly: mean value) required for normalizing continuous variables

    stat_lst =[]
    for col in col_lst:
        stat_lst.append([df[col].mean(), df[col].std()])
    stat_dict = dict(zip(col_lst,stat_lst ))
    return stat_dict

def add_noise(sed, mean, std_per, var):
    # Description: A function to create synthetic noises that will be later to the synthetic value to simulate the errors
    # or the noises that accompany real observations

    # Inputs:
    # sed    : random seed to reproduce
    # mean   : The mean of added noises
    # std_per:
    # var    : variable array to which noises will be added

    std = std_per * var.mean()
    np.random.seed(sed)
    noise = np.random.normal(loc = mean, scale = std, size = len(var))
    var = var + noise
    return var

def cal_stats(pred, obs):
    # Description: A function to create dictionary that has different statistical metrics such as
    # COR : Pearson's Correlation Coefficient
    # RMSE: Root mean square error
    # BIAS: Bias
    # NSE : Nash Sutcliffe Efficiency

    # Inputs:
    # pred  : Predictions
    # obs   : Observations

    # Outputs:
    # stats  : dictionary with the statistical metrics

    COR , _ = pearsonr(pred, obs)
    RMS  = np.sqrt(np.nanmean((pred- obs) ** 2))
    Bias = np.nanmean(pred- obs)
    NSE  = nse(pred, obs)
    stats = {'COR': COR, 'RMS': RMS, 'Bias': Bias, 'NSE': NSE}
    return stats
