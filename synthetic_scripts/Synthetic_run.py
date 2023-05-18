
# THIS FILE IS TO PERFORM THE SYNTHETIC RUNS FOR BOTH SINGLE AND DUAL PARAMETER RECOVERY CASES. THIS FILE SHOULD BE
# RUN AFTER CREATING THE SYNTHETIC DATASETS USING SCRIPT "Forward_run.csv". CHOOSING OPTION:
# 0 ---> SINGLE PARAMETER RECOVERY (Vcmax25_only)
# 1 ---> DUAL PARAMETER RECOVERY (Vcmax25 + B)

# Vcmax25 : The maximum carboxylation rate at 25 C
# B       : A parameter used to compute the soil water stress function btran in the photosynthesis model
########################################################################################################################

#################################
### IMPORT REQUIRED FUNCTIONS ###
#################################
import os
import pandas as pd
import torch.nn as nn

from auxiliary_fn.util           import readfile, onehot_PFT
from auxiliary_fn.predict        import get_data, predict
from auxiliary_fn.Temporal_split import create_sets_temporal_a
from auxiliary_fn.NN_models      import Network
from auxiliary_fn.Stat           import add_noise, cal_stats
from auxiliary_fn.NN_input       import *
from auxiliary_fn.Train          import Train_Vonly, TrainV_B
from plot.util                   import plt_Vretrieval_synthetic, plt_VBretrieval_synthetic

########################################################################################################################

##############################
### DEFINE REQUIRED INPUTS ###
##############################
# 1. option = 0 ---> Single parameter recovery
# 2. option = 1 ---> Dual parameter recovery
option = 1

seed  = 55


### Define the working device ###
if torch.cuda.is_available():
    dev = 7
    torch.cuda.set_device(dev)
else:
    dev = 'cpu'

### Define the synthetic experiment names ###
# First experiment : Single parameter recovery
# Second experiment: Dual parameter recovery
Expnm_lst  = ['Synthetic_OnePAR','Synthetic_TwoPAR']

### Define the dataset csv psths for the first and second experiment###
df_path_lst = ["/data/Synthetic_case/One_parameter/Synthetic_Data_OnePAR.csv",
               "/data/Synthetic_case/Two_parameter/Synthetic_Data_TwoPAR.csv"]

### Define the output directories for the first and second experiment###
out_dir_lst= ["/Results/Synthetic_case/One_parameter","/Results/Synthetic_case/Two_parameter"]

### Define the number of epcochs for the first and second experiment###
Epochs_lst = [100,600]

### Define the list of plant functional types in the dataset ###
pft_lst    = ['Crop R', 'NET Boreal', 'BET Tropical', 'NET Temperate',
             'BET Temperate', 'BDT Temperate', 'C3 grass', 'BDS Temperate','C4 grass']

### Define the target column ###
# should be the same as the one defined in 'Forward_run.py' script
target_col = 'Photo_synthetic'

### Define the synthetic coefficients to compute B_synthetic (summation must be 1)###
OM_frac   =0.1 ; # coefficient for the organic matter fraction
sand_frac =0.45; # coefficient for the sand fraction
clay_frac =0.45; # coefficient for the clay fraction

### Define the minimum and the maximum ranges for the maximum carboxylation rate 25 C to be used to scale the neural
### network output, these values must cover the range of Vcmax25 used to create the synthetic datasets
vcmax_max = 150
vcmax_min = 20

### Define the percentage of the dataset to be used for training ###
trn_ratio = 0.8

########################################################################################################################

##############################
###       MODEL RUN        ###
##############################

# nly      : Number of soil layers, only one used for the synthetic experiments
nly = 1

# Switch variable to to determine whether to just use the defaults (False) or to train a neural network to recover B values (True)
# should be True for both single and dual parameter recovery experiments
V_trained = True

# Switch variable to determine whether to just use the defaults (False) or to train a neural network to recover B values (True)
#                : False ----> single parameter recovery
#                : True  ----> Dual parameter recovery
B_trained = False if option ==  0 else True   # V_trained: a switch to determine whether


### Define the following after slecting option 0 or 1:
# Expname         : The experiment name
# df_path         : The path to the dataset dataframe
# output_directory: The output directory
# Epochs          : The number of epochs
Expname   = Expnm_lst[option]; df_path= df_path_lst[option]; output_directory = out_dir_lst[option];Epochs = Epochs_lst[option]


### read the dataset using read file function ###
# 'readfile' function requires three arguments as the following: readfile(csv_file path, pft_lst, to_frac)
# csv_file path: path for the dataset csv file
# pft_lst      : list of plant functional types in the dataset
# to_Frac      : a switch to convert RH%, sand%, and clay% to fractions if available as percentages

#Be careful whether:
# True (RH, soil clay and sand are not fractions in the synthetic dataset)
# or False (RH, soil clay and sand are already fractions in the synthetic dataset)
df_input_all = readfile(df_path, pft_lst, to_frac = False)
# Add user-defined noise to the synthetic net photosynthetic rates to resemble inaccuracies in the real measurements
df_input_all[target_col] = add_noise(sed=42, mean = 0.0,std_per=0.05, var = df_input_all[target_col].values)
# perform the one-hot encoding step to prepare inputs for the Vcmax25 neural network
df_input_all = onehot_PFT(df_input_all, pft_lst)



### Split the datasdet into training , testing
train, test  = create_sets_temporal_a(df_input_all, ratio = trn_ratio)



### get the forcing variables for the training dataset required for predict function ###
forcing_train = get_data(train)
# transfer to the working device (essential if using GPU)
forcing_train_dev = []
for i in range(len(forcing_train)):
    forcing_train_dev.append(forcing_train[i].to(dev))
forcing_train_dev = tuple(forcing_train_dev)


### get the forcing variables for the testing dataset required for predict function ###
forcing_tst = get_data(test)
# transfer to the working device (essential if using GPU)
forcing_tst_dev = []
for i in range(len(forcing_tst)):
    forcing_tst_dev.append(forcing_tst[i].to(dev))
forcing_tst_dev = tuple(forcing_tst_dev)

### Define the train and test target (here net photosynthetic rate) and the criterion
Anobs_trn = torch.tensor(train[target_col].values, dtype = torch.float, device=dev)
Anobs_tst = torch.tensor(test[target_col].values, dtype = torch.float, device=dev)
criterion=nn.MSELoss()

### Vcmax25 ###
if V_trained:
    Vcat_trn = V_input(train, dev, pft_lst)
    Vcat_tst = V_input(test, dev, pft_lst)

    # Intialize NN_v: a neural network to predict Vc,max25 values
    torch.manual_seed(seed)
    NN_v = Network(len(pft_lst), len(pft_lst), 1); NN_v = NN_v.to(dev)
else:
    # Use defaults from CLM4.5
    Vcmax25_trn = torch.tensor(train.vcmax_CLM45.values,dtype = torch.float, device = dev)
    Vcmax25_tst = torch.tensor(test.vcmax_CLM45.values, dtype=torch.float, device=dev)

### B and btran ###
if B_trained:
    _, Bcont_train   = B_input([], train, nly, dev)
    _, Bcont_tst     = B_input([], test, nly, dev)
    btran_params_trn = get_btran_params(train, dev, nly)
    btran_params_tst = get_btran_params(test, dev, nly)

    # Intialize NN_b: a neural network to predict parameter B values
    torch.manual_seed(seed)
    NN_b = Network(In=Bcont_train.shape[1], nh =Bcont_train.shape[1], out = 1); NN_b = NN_b.to(dev)
else:
    # Use a constant value for btran
    btran_trn = 1.0
    btran_tst = 1.0

### Train the neural networks ###
# 1. option = 0 ---> Single parameter recovery
# Train only NN_v model to recover Vcmax25
if option == 0:
    NN_v = Train_Vonly(epochs    = Epochs ,
                       Vcat      = Vcat_trn,
                       target    = Anobs_trn,
                       vcmax_max = vcmax_max,
                       vcmax_min = vcmax_min,
                       NN_v      = NN_v,
                       forcing   = forcing_train_dev,
                       btran     = btran_trn,
                       criterion = criterion)

# 2. option = 1 ---> Dual parameter recovery
# Train  NN_v and NN_b models to recover Vcmax25 and B respectively
elif option ==1:

    torch.manual_seed(seed)
    NN_v = Network(len(pft_lst), len(pft_lst), 1); NN_v = NN_v.to(dev)
    NN_b = Network(In=Bcont_train.shape[1], nh =Bcont_train.shape[1], out = 1); NN_b = NN_b.to(dev)
    NN_v, NN_b = TrainV_B(epochs = Epochs,
                          Vcat   = Vcat_trn,
                          Bcat   = None,
                          Bcont  = Bcont_train,
                          btran_params = btran_params_trn,
                          target = Anobs_trn,
                          NN_v   = NN_v,
                          NN_b   = NN_b,
                          forcing= forcing_train_dev,
                          criterion = criterion,
                          nly    = nly,
                          vcmax_max = vcmax_max, vcmax_min = vcmax_min)


### Evaluate the trained models for Vcmax25 and B
NN_v.eval() if V_trained else None
NN_b.eval() if B_trained else None

if V_trained:
    with torch.no_grad():
        Vcmax25_trn = NN_v(Vcat_trn).detach().reshape(-1)
        Vcmax25_trn = Vcmax25_trn * (vcmax_max - vcmax_min) + vcmax_min

        Vcmax25_tst = NN_v(Vcat_tst).detach().reshape(-1)
        Vcmax25_tst = Vcmax25_tst * (vcmax_max - vcmax_min) + vcmax_min

if B_trained:

    with torch.no_grad():
        B_trn     = NN_b(Bcont_train).detach().reshape(-1)
        btran_trn = get_btran(btran_params_trn, B_trn, nly)

        B_tst     = NN_b(Bcont_tst).detach().reshape(-1)
        btran_tst = get_btran(btran_params_tst, B_tst, nly)

########################################################################################################################

### compute the simulated net photosynthetic rates using predict function for the training dataset ###
# 'predict' function (see auxiliary_fn.predict) requires three arguments as the following: predict(data, pp1, pp2)
# data: Forcing variables
# pp1 : First parameter , here "Vcmax25"
# pp2 : Second parameter, her "B"
Anpred_trn, ftol_train = predict(forcing_train_dev, Vcmax25_trn, btran_trn)
Anpred_trn = Anpred_trn.detach().to('cpu').numpy()
Anobs_trn  = Anobs_trn.detach().to('cpu').numpy()
# Compute the performance metrics
stats_trn = cal_stats(Anpred_trn, Anobs_trn)
train['Vcmax25_recovered'] = Vcmax25_trn.to('cpu')


### compute the simulated net photosynthetic rates using predict function for the testing dataset ###
Anpred_tst, ftol_tst = predict(forcing_tst_dev,Vcmax25_tst,btran_tst)
Anpred_tst = Anpred_tst.detach().to('cpu').numpy()
Anobs_tst = Anobs_tst.detach().to('cpu').numpy()
# Compute the performance metrics
stats_tst = cal_stats(Anpred_tst, Anobs_tst)
test['Vcmax25_recovered'] = Vcmax25_tst.to('cpu')

########################################################################################################################

### compute the performance metrics
Results_metrics= pd.DataFrame()
Results_metrics['An_trn'] = stats_trn
Results_metrics['An_tst'] = stats_tst


out_path = os.path.join(output_directory,"Results_metrics_{}.csv".format(Expname))
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
Results_metrics.to_csv(out_path)

########################################################################################################################
### Make plots ###
data = pd.concat([train['PFT'], train['vcmax_CLM45'], train['Vcmax25_recovered']], axis=1); data = data.drop_duplicates()
PFT_uni     = data.PFT
vcmax_CLM45 = data.vcmax_CLM45
vcmax_sim   = data.Vcmax25_recovered

### Make plots for retrieval of Vcmax25, Anet ###
if option == 0:
    # Vcmax25
    plt_Vretrieval_synthetic(PFT_uni, output_directory, Expname,
                             Anobs_trn, Anobs_tst,
                             Anpred_trn, Anpred_tst,
                             vcmax_CLM45, vcmax_sim)

### Make plots for retrieval of Vcmax25, B, btran, and Anet ###
else:

    # calculate synthetic b, espi, btran values
    B_trn_syn, epsi_trn_syn, btran_trn_syn = get_btran_synthetic(train, OM_frac, sand_frac, clay_frac, nly)
    B_tst_syn, epsi_tst_syn, btran_tst_syn = get_btran_synthetic(test , OM_frac, sand_frac, clay_frac, nly)

    plt_VBretrieval_synthetic(PFT_uni        ,output_directory,Expname,
                              btran_trn      ,btran_tst       ,btran_trn_syn , btran_tst_syn,
                              B_trn          ,B_tst           ,B_trn_syn     , B_tst_syn    ,
                              Anobs_trn      , Anobs_tst      ,Anpred_trn    , Anpred_tst   ,
                              vcmax_CLM45    , vcmax_sim)




