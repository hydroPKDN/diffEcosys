###
# THIS FILE IS TO PERFORM THE SYNTHETIC RUNS FOR BOTH SINGLE AND DUAL PARAMETER RECOVERY CASES. THIS FILE SHOULD BE
# RUN AFTER CREATING THE SYNTHETIC DATASETS USING SCRIPT "Forward_run.csv". CHOOSING OPTION:
# 0 ---> SINGLE PARAMETER RECOVERY (V_only)
# 1 ---> DUAL PARAMETER RECOVERY (V + B)
###

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

#################################################################################################################
#0_Retrieve V, 1_Retrieve V&B
option    = 0

#REQUIRED INPUTS
seed  = 55
gpuid = 4
torch.cuda.set_device(gpuid)
Expnm_lst  = ['Synthetic_OnePAR','Synthetic_TwoPAR']

df_dir_lst = ["/data/Synthetic_case/One_parameter/Synthetic_Data_OnePAR.csv","/data/Synthetic_case/Two_parameter/Synthetic_Data_TwoPAR.csv"]

out_dir_lst= ["/Results/Synthetic_case/One_parameter","/Results/Synthetic_case/Two_parameter"]
Epochs_lst = [100,600]

pft_lst    = ['Crop R', 'NET Boreal', 'BET Tropical', 'NET Temperate',
             'BET Temperate', 'BDT Temperate', 'C3 grass', 'BDS Temperate','C4 grass']
cat_col    = []
target_col = 'Photo_synthetic'

#NUmber of soil layers to use and Synthetic fractions for soil attributes
nly = 1
OM_frac=0.1; sand_frac=0.45; clay_frac=0.45 #[organic matter, %sand, %clay fractions respectively]


vcmax_max = 150
vcmax_min = 20

#train_ratio
trn_ratio = 0.8
#################################################################################################################
Expname   = Expnm_lst[option]; df_directory = df_dir_lst[option]; output_directory = out_dir_lst[option];Epochs = Epochs_lst[option]
V_trained = True
B_trained = False if option ==  0 else True

df_input_all = readfile(df_directory, pft_lst, to_frac = False) #Be careful whether:
                                                                # True (RH, soil clay and sand are not fractions in the synthetic dataset)
                                                                # or False (RH, soil clay and sand are already fractions in the synthetic dataset)
df_input_all[target_col] = add_noise(sed=42, mean = 0.0,std_per=0.05, var = df_input_all[target_col].values)

# One hot encoding for vcmax25 NN
df_input_all = onehot_PFT(df_input_all, pft_lst)

# Train/Test split
train, test  = create_sets_temporal_a(df_input_all, ratio = trn_ratio)

#Forcing data for training set
forcing_train = get_data(train)
forcing_train_gpu = []
for i in range(len(forcing_train)):
    forcing_train_gpu.append(forcing_train[i].to(gpuid))
forcing_train_gpu = tuple(forcing_train_gpu)

#Forcing data for testing set
forcing_tst = get_data(test)
forcing_tst_gpu = []
for i in range(len(forcing_tst)):
    forcing_tst_gpu.append(forcing_tst[i].to(gpuid))
forcing_tst_gpu = tuple(forcing_tst_gpu)

#Define the target (Net photosynthetic rate and the criterion
Anobs_trn = torch.tensor(train[target_col].values, dtype = torch.float, device=gpuid)
Anobs_tst = torch.tensor(test[target_col].values, dtype = torch.float, device=gpuid)
criterion=nn.MSELoss()

# Definition of Vcmax25
if V_trained:
    Vcat_trn = V_input(train, gpuid, pft_lst)
    Vcat_tst = V_input(test, gpuid, pft_lst)
    torch.manual_seed(seed)
    NN_v = Network(len(pft_lst), len(pft_lst), 1); NN_v = NN_v.to(gpuid)
else:
    Vcmax25_trn = torch.tensor(train.vcmax_CLM45.values,dtype = torch.float, device = gpuid)
    Vcmax25_tst = torch.tensor(test.vcmax_CLM45.values, dtype=torch.float, device=gpuid)

# Definition of B
if B_trained:
    _, Bcont_train   = B_input(cat_col, train, nly, gpuid)
    _, Bcont_tst     = B_input(cat_col, test, nly, gpuid)
    btran_params_trn = get_btran_params(train, gpuid, nly)
    btran_params_tst = get_btran_params(test, gpuid, nly)

    torch.manual_seed(seed)
    NN_b = Network(In=Bcont_train.shape[1], nh =Bcont_train.shape[1], out = 1); NN_b = NN_b.to(gpuid)
else:
    btran_trn = 1.0
    btran_tst = 1.0


if option == 0:
    NN_v = Train_Vonly(epochs    = Epochs ,
                       Vcat      = Vcat_trn,
                       target    = Anobs_trn,
                       vcmax_max = vcmax_max,
                       vcmax_min = vcmax_min,
                       NN_v      = NN_v,
                       forcing   = forcing_train_gpu,
                       btran     = btran_trn,
                       criterion = criterion)

elif option ==1:
    #
    torch.manual_seed(seed)
    NN_v = Network(len(pft_lst), len(pft_lst), 1); NN_v = NN_v.to(gpuid)
    NN_b = Network(In=Bcont_train.shape[1], nh =Bcont_train.shape[1], out = 1); NN_b = NN_b.to(gpuid)
    NN_v, NN_b = TrainV_B(epochs = Epochs,
                          Vcat   = Vcat_trn,
                          Bcat   = None,
                          Bcont  = Bcont_train,
                          btran_params = btran_params_trn,
                          target = Anobs_trn,
                          NN_v   = NN_v,
                          NN_b   = NN_b,
                          forcing= forcing_train_gpu,
                          criterion = criterion,
                          nly    = nly,
                          vcmax_max = vcmax_max, vcmax_min = vcmax_min)


#Model evaluation
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

##############################################################################################################TRAIN
Anpred_trn, ftol_train = predict(forcing_train_gpu, Vcmax25_trn, btran_trn)
Anpred_trn = Anpred_trn.detach().to('cpu').numpy()
Anobs_trn  = Anobs_trn.detach().to('cpu').numpy()

stats_trn = cal_stats(Anpred_trn, Anobs_trn)
train['Vcmax25_recovered'] = Vcmax25_trn.to('cpu')
#############################################################################################################TEST
Anpred_tst, ftol_tst = predict(forcing_tst_gpu,Vcmax25_tst,btran_tst)
Anpred_tst = Anpred_tst.detach().to('cpu').numpy()
Anobs_tst = Anobs_tst.detach().to('cpu').numpy()

stats_tst = cal_stats(Anpred_tst, Anobs_tst)
test['Vcmax25_recovered'] = Vcmax25_tst.to('cpu')
#############################################################################################################
# Results
Results_metrics= pd.DataFrame()
Results_metrics['An_trn'] = stats_trn
Results_metrics['An_tst'] = stats_tst
print(Results_metrics)

Results_metrics.to_csv(os.path.join(output_directory,"Results_metrics_{}.csv".format(Expname)))

#plotting
data = pd.concat([train['PFT'], train['vcmax_CLM45'], train['Vcmax25_recovered']], axis=1); data = data.drop_duplicates()
PFT_uni     = data.PFT
vcmax_CLM45 = data.vcmax_CLM45
vcmax_sim   = data.Vcmax25_recovered

if option == 0:
    # Vcmax25
    plt_Vretrieval_synthetic(PFT_uni, output_directory, Expname,
                             Anobs_trn, Anobs_tst,
                             Anpred_trn, Anpred_tst,
                             vcmax_CLM45, vcmax_sim)

else:

    # calculate synthetic b, espi, btran values
    B_trn_syn, epsi_trn_syn, btran_trn_syn = get_btran_synthetic(train, OM_frac, sand_frac, clay_frac, nly)
    B_tst_syn, epsi_tst_syn, btran_tst_syn = get_btran_synthetic(test , OM_frac, sand_frac, clay_frac, nly)

    plt_VBretrieval_synthetic(PFT_uni        ,output_directory,Expname,
                              btran_trn      ,btran_tst       ,btran_trn_syn , btran_tst_syn,
                              B_trn          ,B_tst           ,B_trn_syn     , B_tst_syn    ,
                              Anobs_trn      , Anobs_tst      ,Anpred_trn    , Anpred_tst   ,
                              vcmax_CLM45    , vcmax_sim)




