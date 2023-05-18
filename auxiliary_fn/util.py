import pandas as pd
import numpy as np

########################################################################################################################
def readfile(file, pft_lst, to_frac = True):
    # Description: A function to read the csv file including the  dataset

    # Inputs:
    # file   : csv file to be read
    # pft_lst: list of the plant functional type categories to be considered for a specific model run
    # to_frac: Whether to convert the sand percentage , clay percentage and relative humidity to fractions or not.
    # This argumnent is very tricky so needs to make sure if they were already converted to fractions in your dataaset or not.
    # Sand percentage and clay percentage need to be fractions to be used inside the B neural network
    # Also RH needs to be converted to fraction (see Functions_FatesModel.py)

    # Output:
    # df    : The dataset dataframe

    # read and keep data points corresponding to PFT_lst
    pft_lst = list(np.sort(pft_lst))
    df = pd.read_csv(file)
    df = df[df['PFT'].isin(pft_lst)]
    df = df.reset_index(drop = 'True')

    # approximate all Vcmax_CLM45 to two decimal places
    df['vcmax_CLM45'] = round(df['vcmax_CLM45'],2)

    # convert all percentage data to fraction
    if to_frac:
        colnames = df.columns.tolist()
        subs = ['soil_sand', 'soil_clay', 'RH']
        subcols = []
        for sub in subs:
            subcols = subcols + [i for i in colnames if sub in i]
        df[subcols] = df[subcols]/100
    return df
########################################################################################################################
def onehot_PFT(df, pft_lst):
    # Description: A function to one hot encode the Plant functional type (PFT) column in the dataset
    # Inputs :
    # df     : The dataset dataframe
    # pft_lst: list of the plant functional type categories to be considered for a specific model run

    one_hot_PFT = pd.get_dummies(df['PFT'])
    df          = df.reset_index(drop = True);
    one_hot_PFT = one_hot_PFT.reset_index(drop = True)
    df          = df.join(one_hot_PFT)
    # if isinstance(colname, list):
    #     for cat in colname:
    #         df[cat] = df[cat].astype('category')
    #         df[cat] = df[cat].cat.reorder_categories(pft_lst)
    # else:
    df['PFT'] = df['PFT'].astype('category')
    df['PFT'] = df['PFT'].cat.reorder_categories(pft_lst)
    return df

########################################################################################################################

