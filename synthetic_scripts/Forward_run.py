# THIS SCRIPT MUST BE RUN FIRST BEFORE RUNNING THE SCRIPT CALLED 'Synthetic_run.py' TO GENERATE THE SYNTHETIC DATASET

# THIS SCRIPT CAN BE MANILY USED TO GENERATE SYNTHETIC DATA FOR PHOTOSYNTHETIC RATES TO BE USED IN THE SYNTHETIC
# EXPERIMENTS

# IN THIS PROJECT, WE RUN TWO SYNTHETIC EXPERIMENTS ONE FOR SINGLE PARAMETER RECOVERY (V_cmax25) ONLY AND ANOTHER FOR
# DUAL PARAMETER RECOVERY (V_cmax25 - B)
# V_cmax25 : The maximum carboxylation rate at 25 C
# B        : A parameter used to compute the soil water stress function btran for the photosynthetic model

# TO PRODUCE SYNTHETIC DATA FOR THE SINGLE PARAMETER RECOVERY CASE:
# 1.THE FILE NAME WOULD BE "Synthetic_Data_OnePAR.csv"
# 2. B_synthesize = False

# TO PRODUCE SYNTHETIC DATA FOR THE DUAL PARAMETER RECOVERY CASE:
# 1.THE FILE NAME WOULD BE "Synthetic_Data_TwoPAR.csv"
# 2. B_synthesize = True

# Note: A tricky point, since here when reading the file, we keep (to_frac = True), therefore,
# when using the output file (which has the percentages converted to fractions),
# the argument (to_frac should be False) as in the script named "Synthetic_run.py"
########################################################################################################################

#################################
### IMPORT REQUIRED FUNCTIONS ###
#################################
import os
import numpy as np
import torch.cuda

from auxiliary_fn.util     import readfile
from auxiliary_fn.newton   import *
from auxiliary_fn.predict  import predict
from auxiliary_fn.NN_input import get_btran_synthetic

########################################################################################################################

##############################
### DEFINE REQUIRED INPUTS ###
##############################
# 1. option = 0 ---> Single parameter recovery
# 2. option = 1 ---> Dual parameter recovery
option = 1

### Define the working device ###
if torch.cuda.is_available():
    dev = 7
    torch.cuda.set_device(dev)
else:
    dev = 'cpu'

### Define the dataset directory ###
df_directory = "/data/example_dataset"

### Define the list of plant functional types in the dataset ###
pft_lst      = ['Crop R'       , 'NET Boreal'   , 'BET Tropical' ,
                'NET Temperate','BET Temperate' , 'BDT Temperate',
                'C3 grass'     , 'BDS Temperate', 'C4 grass'
                ]
### Define the output directory ###
out_dir      = "/data/Synthetic_case/"

### Define the synthetic coefficients to compute B_synthetic (summation must be 1)###
OM_frac   =0.1 ; # coefficient for the organic matter fraction
sand_frac =0.45; # coefficient for the sand fraction
clay_frac =0.45; # coefficient for the clay fraction


########################################################################################################################

##############################
###       MODEL RUN        ###
##############################

# Switch variable: False ----> single parameter recovery
#                : True  ----> Dual parameter recovery
B_synthesize = False if option == 0 else True

### read the dataset using read file function ###
# 'readfile' function requires three arguments as the following: readfile(csv_file path, pft_lst, to_frac)
# csv_file path: path for the dataset csv file
# pft_lst      : list of plant functional types in the dataset
# to_Frac      : a switch to convert RH%, sand%, and clay% to fractions if available as percentages
df_input      = readfile(os.path.join(df_directory, 'example_dataset.csv'), pft_lst= pft_lst, to_frac= True)

### Define Vcmax25 ###
Vcmax25       = torch.tensor(df_input.vcmax_CLM45.values,dtype = torch.float, device = dev)

### Define btran ###
if B_synthesize:
    B, epsi, btran= get_btran_synthetic(df_input, OM_frac, sand_frac, clay_frac, nly = 1)
    btran         = torch.tensor(btran,dtype = torch.float, device = dev)
else:
    btran = torch.tensor(1.0, device=dev)

### get the forcing variables required for predict function ###
forcing     = get_data(df_input)

# transfer to the working device (essential if using GPU)
forcing_dev = []
for i in range(len(forcing)):
    forcing_dev.append(forcing[i].to(dev))
forcing_dev = tuple(forcing_dev)

### compute the synthetic net photosynthetic rates using predict function ###
# 'predict' function (see auxiliary_fn.predict) requires three arguments as the following: predict(data, pp1, pp2)
# data: Forcing variables
# pp1 : First parameter , here "Vcmax25"
# pp2 : Second parameter, her "B"
An_sim = np.array(predict(forcing_dev, Vcmax25, btran)[0].detach().to('cpu'));
df_input['Photo_synthetic'] = An_sim


### output the synthetic dataset created to be used for synthetic runs ###
# Single parameter
if option == 0:
    out_path = os.path.join(out_dir, 'One_parameter')
    out_file = os.path.join(out_path, 'Synthetic_Data_OnePAR.csv')
# Dual parameter
elif option == 1:
    out_path = os.path.join(out_dir, 'Two_parameter')
    out_file = os.path.join(out_path, 'Synthetic_Data_TwoPAR.csv')

if not os.path.exists(out_path):
    os.makedirs(out_path)

df_input.to_csv(out_file, index = False)


