###
# THIS FILE IS TO CREATE DIFFERENT FORMS OF DATASET SPLITTING
# 1.TEMPORALLY (OLDER DATES FOR TRAINING AND NEWER DATES FOR TESTING)
# 2. K-FOLDS TO PERFORM CROSS VALIDATION TEST

import pandas as pd
import numpy as np

#######################################################################################################################
def create_sets_temporal_a(df_all, ratio):
    # Description: A function to split the dataset temporally with train = ratio * size of dataset
    # Inputs:
    # df    : The dataframe dataset
    # ratio : percentage of the dataset used for training

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all         = df_all.sort_values(by = ['Location','Date'])
    df_all         = df_all.reset_index(drop = True)

    train_set = pd.DataFrame()
    test_set  = pd.DataFrame()

    # First location layer
    for loc in df_all.Location.unique():
        df_loc = df_all[df_all['Location'] == loc]

        # Second PFT Layer
        for pft in df_loc.PFT.unique():
            df_temp = df_loc[df_loc['PFT'] == pft]

            # Number of dates of measurements available for a specific PFT in a specific location
            n       = len(df_temp['Date'].unique())
            if n == 1:
                train_set= pd.concat([train_set,df_temp])
            else:
                ntrain    = np.int(np.floor(ratio * n))
                All_dates = np.sort(df_temp['Date'].unique())
                train_date= All_dates[:ntrain]
                df_train  = df_temp[df_temp['Date'].isin(train_date)]
                df_test   = df_temp.drop(df_train.index)
                train_set = pd.concat([train_set,df_train])
                test_set  = pd.concat([test_set ,df_test])

    train_set = train_set.reset_index(drop=True)
    test_set  = test_set.reset_index(drop=True)

    return train_set, test_set

#######################################################################################################################
def create_FIVEfolds(df_all, sed):

    # Description: A function to split the dataset into 5 folds for implementing the cross validation
    # Inputs:
    # df    : The dataframe dataset
    # sed   : The random seed to reproduce the folds

    # k represents the number of folds
    k = 5

    # varies with k
    ratios = [.20,0.25,1/3.0, 0.5]
    train_sets = [pd.DataFrame() for i in range(k)]
    test_sets  = [pd.DataFrame() for i in range(k)]
    wholeset   = df_all
    #wholeset  = shuffle(wholeset, random_state=sed)
    for i in range(k-1):

        # First location layer
        for loc in wholeset.Location.unique():
            df_loc = wholeset[wholeset['Location'] == loc]

            # Second PFT Layer
            for pft in df_loc.PFT.unique():
                df_temp = df_loc[df_loc['PFT'] == pft]

                # length of dataset with specific PFT in a specific location
                n       = len(df_temp)
                if n == 1:
                    train_sets[i] = pd.concat([train_sets[i],df_temp])
                else:
                    df_train      = df_temp.sample(n = np.int(np.floor((1 - ratios[i]) * len(df_temp))), random_state=sed)
                    df_test       = df_temp.drop(df_train.index)
                    train_sets[i] = pd.concat([train_sets[i],df_train])
                    test_sets[i]  = pd.concat([test_sets[i],df_test])

        wholeset      = wholeset.drop(test_sets[i].index)
        train_sets[i] = df_all.drop(test_sets[i].index)

    test_sets[-1] = wholeset
    train_sets[-1]= df_all.drop(test_sets[-1].index)
    return train_sets, test_sets
#######################################################################################################################
