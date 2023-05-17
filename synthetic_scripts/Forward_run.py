# THIS SCRIPT CAN BE MANILY USED TO GENERATE SYNTHETIC DATA FOR PHOTOSYNTHETIC RATES TO BE USED IN THE SYNTHETIC
# EXPERIMENTS

# IN THIS PROJECT, WE RUN TWO SYNTHETIC EXPERIMENTS FOR SINGLE PARAMETER RECOVERY (V_cmax) ONLY AND DUAL PARAMETER
# RECOVERY (V_cmax - B)

# TO PRODUCE SYNTHETIC DATA FOR THE SINGLE PARAMETER RECOVERY CASE:
# 1.THE FILE NAME WOULD BE "Synthetic_Data_OnePAR.csv"
# 2. B_synthesize = False

# TO PRODUCE SYNTHETIC DATA FOR THE DUAL PARAMETER RECOVERY CASE:
# 1.THE FILE NAME WOULD BE "Synthetic_Data_TwoPAR.csv"
# 2. B_synthesize = True

# Note: A tricky point, since here when reading the file, we keep (to_frac = True), therefore,
# when using the output file (which has the percentages converted to fractions),
# the argument (to_frac should be False) as in the script named "Synthetic_case"

import numpy as np
from auxiliary_fn.util     import readfile
from auxiliary_fn.newton   import *
from auxiliary_fn.predict  import predict
from auxiliary_fn.NN_input import get_btran_synthetic
#################################################################################################################
#REQUIRED INPUTS
gpuid = 4
torch.cuda.set_device(gpuid)
df_directory = "/data/original_dataset.csv"
pft_lst      = ['Crop R', 'NET Boreal', 'BET Tropical', 'NET Temperate',
                'BET Temperate', 'BDT Temperate', 'C3 grass', 'BDS Temperate', 'C4 grass',
                      ]
out_file     = "/data/Synthetic_case/One_parameter/Synthetic_Data_OnePAR.csv"

nly    = 1
OM_frac=0.1; sand_frac=0.45; clay_frac=0.45
B_synthesize = False
#################################################################################################################
# Model run
df_input      = readfile(df_directory, pft_lst= pft_lst, to_frac= True)
Vcmax25       = torch.tensor(df_input.vcmax_CLM45.values,dtype = torch.float, device = gpuid)
if B_synthesize:
    B, epsi, btran= get_btran_synthetic(df_input, OM_frac, sand_frac, clay_frac, nly)
    btran         = torch.tensor(btran,dtype = torch.float, device = gpuid)
else:
    btran = torch.tensor(1.0, device=gpuid)

forcing = get_data(df_input)
forcing_gpu = []
for i in range(len(forcing)):
    forcing_gpu.append(forcing[i].to(gpuid))
forcing_gpu = tuple(forcing_gpu)

An_sim = np.array(predict(forcing_gpu, Vcmax25, btran)[0].detach().to('cpu'))
df_input['Photo_synthetic'] = An_sim
df_input.to_csv(out_file, index = False)


