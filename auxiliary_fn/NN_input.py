###
# THIS FILE INCLUDES ALL THE FUNCTIONS REQUIRED TO PREPARE THE INPUTS FOR DIFFERENT NEURAL NETWORKS USED IN THIS PROJECT
# INLCUDING NEURAL NETWORKS TO PREDICT THE MAXIMUM CARBOXYLATION RATE AT 25 C AND THE PARAMETER B
###

import numpy as np
import torch
from auxiliary_fn.Stat import normalize_stat


def B_input(cat_cols, df, nly, dev):
    # Description:prepare the categorical and the continous inputs for B Neural Network
    # The function assumes that the continous inputs always include the three soil attributes: percentage of sand (%sand),
    # percentage of clay (%clay) and fraction of organic matter (Fom)

    # Inputs:
    # cat_cols: Categorical columns used for B Neural Network
    # df      : dataframe with the required datasets
    # nly     : number of soil layers considered
    # dev     : the device whether 'cpu' or 'gpu'

    # Outputs:
    # Cats.: categorical inputs
    # Conts: continous inputs

    if cat_cols == []:
        cats = []
    elif isinstance(cat_cols, list):
        cats= np.stack([df[col].cat.codes.values for col in cat_cols], axis = 1)
    else:
        cats= df[cat_cols].cat.codes.values
    cats = torch.tensor(cats, dtype = torch.int32, device=dev)
    cats = cats.repeat(nly,1)
    conts= np.empty((0,3))
    for ly in range(nly):
        cont_cols = [f'soil_clay_{ly}', f'soil_sand_{ly}', f'soil_OM_{ly}']
        conts_lay = np.stack([df[col].values for col in cont_cols], axis = 1)
        conts = np.append(conts, conts_lay, axis = 0)
    conts = torch.tensor(conts, dtype= torch.float, device=dev)

    return cats, conts

def V_input(df, dev, pft_lst):
    # Description: prepare the categorcial input for Vcmax25 Neural Network (in thise case the PFT)

    # Inputs:
    # df     : dataframe with the required datasets
    # dev    : the device whether 'cpu' or 'Cuda'
    # pft_lst: list of plant functional types inlcuded in this dataset

    # Outputs:
    # input: which represent  the categorcial input for Vcmax25 Neural Network (in thise case the PFT)
    input = torch.tensor(df[pft_lst].values, device = dev)
    input = input.float()
    return input

def get_btran_synthetic(df, OM_frac, sand_frac, clay_frac, nly):
    # Description: calculate the synthetic values of btran : soil water stress factor, where btran should be > 0 and < 1

    # Inputs
    # df       : dataframe with the required datasets
    # OM_Frac  : weight of the fraction of the organic matter in the computation of B parameter
    # sand_frac: weight of the percentage of sand in the computation of B parameter
    # clay_frac: weight of the percentage of clay in the computation of B parameter
    # nly      : number of soil layers considered

    # Outputs:
    # B    : synthetic values for parameter B
    # epsi : soil matric potential for all soil layers
    # btran: synthetic values for the soil water stress factor (btran)


    # soil matric potential for open stomata
    epsi_o = df.epsi_o.values
    # soil matric potential for closed stomata
    epsi_c = df.epsi_c.values

    # Initialize btran, B , and epsi
    btran  = 0.0
    B      = np.empty((0))
    epsi   = np.empty((0))

    # loop through soil layers
    for ly in range(nly):
        # Fraction of organic matter
        soil_OM_i   = df[f'soil_OM_{ly}'].values
        # sand percentage
        soil_sand_i = df[f'soil_sand_{ly}'].values
        # clay percentage
        soil_clay_i = df[f'soil_clay_{ly}'].values
        # calculate synthetic B for layer i as:
        B_i         = OM_frac * soil_OM_i + sand_frac * soil_sand_i + clay_frac* soil_clay_i
        # soil wettness for layer i
        SMC_i       = df[f'SMC_{ly}'].values
        # soil water potential for layer i
        epsi_i      = np.maximum(epsi_o * SMC_i ** -B_i,epsi_c)
        if nly == 1:
            r_i = 1.0
        else:
            # plant root distribution for layer i
            r_i     = df[f'r{ly+1}'].values
        # plant wilting factor for layer i
        w_i         = np.minimum((epsi_c - epsi_i) / (epsi_c - epsi_o), 1.0)
        # Accumulate btran across different soil layers
        btran      +=  r_i * w_i

        # Append B and epsi across different soil layers
        B = np.append(B, B_i)
        epsi = np.append(epsi, epsi_i)

    # ensure 0.0 <= btran <= 1.0
    btran = np.maximum(btran,0.0)

    return B, epsi, btran

def get_btran(params, B, nly):
    # Description: calculate btran : soil water stress factor, where btran should be > 0 and < 1
    # using the B values learned by NN_B

    # Inputs:
    # params: parameters used in btran calculations from "get_btran_params" function
    # B     : parameter B values learnt by B neural network
    # nly   : number of soil layers considered

    # Outputs:
    # btran: The soil water stress factor (btran) based on learned B values by NN_B

    r, SMC, epsi_c, epsi_o = params
    r = [1.0] if nly == 1 else r

    epsi = torch.max(epsi_o.repeat(nly) * SMC.view(-1) ** -B, epsi_c.repeat(nly))
    epsi = epsi.view(nly, -1)

    btran = r[0] * torch.clamp((epsi_c - epsi[0]) / (epsi_c - epsi_o),max = 1.0)
    for ly in range(1, nly):
        btran += r[ly] * torch.clamp((epsi_c - epsi[ly]) / (epsi_c - epsi_o),max = 1.0)
    btran = torch.clamp(btran, min = 0.0)

    return btran

def get_btran_params(df, dev, nly):
    # Description: calculate btran : soil water stress factor, where btran should be > 0 and < 1
    # using the B values learned by NN_B

    # Inputs:
    # df    : dataframe with the required datasets
    # dev   : the device whether 'cpu' or 'Cuda'
    # nly   : number of soil layers considered

    # Outputs:
    # parameters used in btran calculations as:
    # r      : plant root distribution based on PFT
    # SMC    : soil wettness
    # epsi_o : soil matric potential for open stomata
    # epsi_c : soil matric potential for closed stomata


    r = np.empty((nly, len(df))); SMC = np.empty((nly, len(df)))
    for ly in range(nly):
        SMC[ly] = df[f'SMC_{ly}'].values
        r[ly]   = df[f'r{ly+1}'].values

    r   = torch.FloatTensor(r).to(dev)
    SMC = torch.FloatTensor(SMC).to(dev)

    epsi_o = torch.FloatTensor(df.epsi_o.values).to(dev)
    epsi_c = torch.FloatTensor(df.epsi_c.values).to(dev)
    return r, SMC, epsi_c, epsi_o

