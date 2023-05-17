###
# THIS FILE INCLUDES ALL THE FUNCTIONS REQUIRED TO GIVE THE FINAL PREDICTIONS OF NET PHOTOSYNTHESIS RATES USING TWO
# STEPS:
# 1. SOLVE THE NONLINEAR SYSTEM FOR THE INTERCELLULAR LEAF CO2 PRESSURE (C_i)
# 2. PREDICT THE NET PHOTOSYNTHESIS RATE AND OTHER VARIABLES USING THE SOLVED FOR (C_i)

from auxiliary_fn.newton import *

######################################################################################################################
def get_guess(c3c4_path_index, can_co2_ppress):
    # Description: A function to get the initial guess for the intercellular leaf CO2 pressure (Ci) which is mainly
    # dependent on the classification of C3 and C4 plants.

    # Inputs:
    # c3c4_path_index: Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
    # can_co2_ppress : Partial pressure of CO2 NEAR the leaf surface (Pa)
    mask = c3c4_path_index == c3_path_index
    mask = mask.type(torch.uint8)
    ci   = (init_a2l_co2_c3 * can_co2_ppress) * mask + (init_a2l_co2_c4 * can_co2_ppress) * (1 - mask)

    return ci
######################################################################################################################
def predict(df,pp1, pp2):
    # Description: The function that predicts the net photosynthetic rate (An) using the learned parameters pp1 and pp2
    # where:

    # Inputs:
    # df : dataframe with the required datasets
    # pp1: The first learned parameter which is here the Vcmax,25 (the plant maximum carboxylation rate at 25 C)
    # pp2: The second parameter which is here btran (soil water stress factor) calculated using learned B values

    # Outputs:
    # anet   : The simulated net photosynthetic rate (umol CO2/m**2/s)
    # ftol   : The solution of the nonlinear system which should be ~ zero >. f(x) = 0.0

    parsun_lsl, c3c4_path_index  , stomatal_intercept_btran_Med, mm_kco2, mm_ko2,\
    co2_cpoint, gb_mol           , ceair                       , lmr    , vcmax ,\
    jmax      , co2_rcurve_islope, medlyn_slope                , can_o2_ppress  ,\
    can_press , can_co2_ppress   , veg_esat                    , qabs           ,\
    je        , LAI              , vpd = pre_fates(df, pp1, pp2)

    # First: Solve the nonlinear system using learned pp1 and pp2 values
    f = Model_equations(df, pp1, pp2)
    J1 = Jacobian(mtd="batchScalarJacobian_AD")#, func=f, settings={"dx":1e-2,"bounds":(10.0, 50.0)}) #mtd="batchScalarJacobian_AD"
    vG = tensorNewton(f,J1)#, mtd="rtsafe",lb=0,ub=1000)
    x0 = get_guess(c3c4_path_index, can_co2_ppress)
    x0.requires_grad_(True)
    x  = vG(x0)
    ftol = f(x)

    # Second: Compute different photosynthetic flux rates
    # ac    : Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
    # aj    : RuBP-limited gross photosynthesis (umol CO2/m**2/s)
    # ap    : product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
    # ai    : intermediate co - limited photosynthesis(umolCO2 / m ** 2 / s)
    # agross: co-limited gross leaf photosynthesis (umol CO2/m**2/s)
    # anet  : net leaf photosynthesis (umol CO2/m**2/s)
    ac = (vcmax * torch.clamp(x - co2_cpoint, min=0.0) / (x + mm_kco2 * (1.0 + can_o2_ppress / mm_ko2))) * c3c4_path_index \
         + vcmax * (1 - c3c4_path_index)

    aj = (je * torch.clamp(x - co2_cpoint, min=0.0) / (4.0 * x + 8.0 * co2_cpoint)) * c3c4_path_index + \
         (quant_eff * parsun_lsl * 4.6) * (1 - c3c4_path_index)

    ap = co2_rcurve_islope * torch.clamp(x, min=0.0) / can_press

    ai = 0.0 * c3c4_path_index + quadratic_min(theta_cj_c4, -(ac + aj), ac * aj) * (1 - c3c4_path_index)

    agross = quadratic_min(theta_cj_c3, -(ac + aj), ac * aj) * c3c4_path_index + \
             quadratic_min(theta_ip, - (ai + ap), ai * ap) * (1 - c3c4_path_index)

    anet   =  agross - lmr

    # Third:  correct anet for LAI > 0.0 and parsun_lsl <=0.0
    ### added them to give the correct values
    mask = LAI > 0.0
    mask = mask.type(torch.uint8)
    anet = mask * anet + (1 - mask) * 0.0

    mask = parsun_lsl <= 0.0
    mask = mask.type(torch.uint8)
    anet = mask * -lmr + (1-mask) * anet

    return anet, ftol

######################################################################################################################
def predict_An_gs(df,pp1, pp2):
    # Description: The function that predicts the net photosynthetic rate (An) and the stomatal conductance (gs_mol)
    # using the learned parameters pp1 and pp2 where:

    # Inputs:
    # df : dataframe with the required datasets
    # pp1: The first learned parameter which is here the Vcmax,25 (the plant maximum carboxylation rate at 25 C)
    # pp2: The second parameter which is here btran (soil water stress factor) calculated using learned B values

    # Outputs:
    # anet   : The simulated net photosynthetic rate (umol CO2/m**2/s)
    # gs_mol : The stomatal conductance (umol H2O/m**2/s)
    # ftol   : The solution of the nonlinear system which should be ~ zero >. f(x) = 0.0

    parsun_lsl, c3c4_path_index  , stomatal_intercept_btran_Med, mm_kco2, mm_ko2,\
    co2_cpoint, gb_mol           , ceair                       , lmr    , vcmax ,\
    jmax      , co2_rcurve_islope, medlyn_slope                , can_o2_ppress  ,\
    can_press , can_co2_ppress   , veg_esat                    , qabs           ,\
    je        , LAI              , vpd = pre_fates(df, pp1, pp2)

    # First: Solve the nonlinear system using learned pp1 and pp2 values
    f = Model_equations(df, pp1, pp2)
    J1 = Jacobian(mtd="batchScalarJacobian_AD")#, func=f, settings={"dx":1e-2,"bounds":(10.0, 50.0)}) #mtd="batchScalarJacobian_AD"
    vG = tensorNewton(f,J1)#, mtd="rtsafe",lb=0,ub=1000)
    x0 = get_guess(c3c4_path_index, can_co2_ppress)
    x0.requires_grad_(True)
    x  = vG(x0)
    ftol = f(x)

    # Second: Compute different photosynthetic flux rates
    # ac    : Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
    # aj    : RuBP-limited gross photosynthesis (umol CO2/m**2/s)
    # ap    : product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
    # ai    : intermediate co - limited photosynthesis(umolCO2 / m ** 2 / s)
    # agross: co-limited gross leaf photosynthesis (umol CO2/m**2/s)
    # anet  : net leaf photosynthesis (umol CO2/m**2/s)
    ac = (vcmax * torch.clamp(x - co2_cpoint, min=0.0) / (x + mm_kco2 * (1.0 + can_o2_ppress / mm_ko2))) * c3c4_path_index \
         + vcmax * (1 - c3c4_path_index)

    aj = (je * torch.clamp(x - co2_cpoint, min=0.0) / (4.0 * x + 8.0 * co2_cpoint)) * c3c4_path_index + \
         (quant_eff * parsun_lsl * 4.6) * (1 - c3c4_path_index)

    ap = co2_rcurve_islope * torch.clamp(x, min=0.0) / can_press

    ai = 0.0 * c3c4_path_index + quadratic_min(theta_cj_c4, -(ac + aj), ac * aj) * (1 - c3c4_path_index)

    agross = quadratic_min(theta_cj_c3, -(ac + aj), ac * aj) * c3c4_path_index + \
             quadratic_min(theta_ip, - (ai + ap), ai * ap) * (1 - c3c4_path_index)

    anet   =  agross - lmr

    # Third:  correct anet for LAI > 0.0 and parsun_lsl <=0.0
    ### added them to give the correct values
    mask = LAI > 0.0
    mask = mask.type(torch.uint8)
    anet = mask * anet + (1 - mask) * 0.0

    mask = parsun_lsl <= 0.0
    mask = mask.type(torch.uint8)
    anet = mask * -lmr + (1-mask) * anet

    # Third: Stomatal Conductance computation
    a_gs = anet
    leaf_co2_ppress = torch.clamp(can_co2_ppress - h2o_co2_bl_diffuse_ratio / gb_mol * a_gs * can_press, min=1.e-06)
    #vpd = torch.clamp((veg_esat - ceair), min=50.0) * 0.001

    term_gsmol = h2o_co2_stoma_diffuse_ratio * anet / (leaf_co2_ppress / can_press)
    aquad = 1.0
    bquad = -(2.0 * (stomatal_intercept_btran_Med + term_gsmol) + (medlyn_slope * term_gsmol) ** 2 / (gb_mol * vpd))
    cquad = stomatal_intercept_btran_Med * stomatal_intercept_btran_Med + \
            (2.0 * stomatal_intercept_btran_Med + term_gsmol * (1.0 - medlyn_slope * medlyn_slope / vpd)) * term_gsmol

    gs_mol = stomatal_intercept_btran_Med.clone()
    mask = (anet < 0.0) | (bquad * bquad < 4.0 * aquad * cquad)  # I assume that complex roots appear only when anet approaches +-zero however this isnot working for the second condition due to nan values
    mask = mask.type(torch.bool)
    gs_mol[torch.logical_not(mask)] = quadratic_max(aquad, bquad[torch.logical_not(mask)],cquad[torch.logical_not(mask)])

    gs_mol = gs_mol/10**6
    return anet, gs_mol, ftol
######################################################################################################################
def predict_Vars(df,pp1, pp2):
    # Description: The function that predicts the net photosynthetic rate (An) and the stomatal conductance (gs_mol)
    # using the learned parameters pp1 and pp2 where:

    # Inputs:
    # df : dataframe with the required datasets
    # pp1: The first learned parameter which is here the Vcmax,25 (the plant maximum carboxylation rate at 25 C)
    # pp2: The second parameter which is here btran (soil water stress factor) calculated using learned B values

    # Outputs:
    # x               : The intercellular leaf CO2 pressure (Ci)
    # leaf_co2_ppress : CO2 partial pressure at leaf surface (Pa)
    # vpd             : Vapor pressure deficit (kpa)
    # anet            : The simulated net photosynthetic rate (umol CO2/m**2/s)
    # gs_mol1 &2      : The stomatal conductance (umol H2O/m**2/s) cpomputed using differetn equations
    # ac, aj, ap      : defined afterwards

    parsun_lsl, c3c4_path_index  , stomatal_intercept_btran_Med, mm_kco2, mm_ko2,\
    co2_cpoint, gb_mol           , ceair                       , lmr    , vcmax ,\
    jmax      , co2_rcurve_islope, medlyn_slope                , can_o2_ppress  ,\
    can_press , can_co2_ppress   , veg_esat                    , qabs           ,\
    je        , LAI              , vpd= pre_fates(df, pp1, pp2)

    # First: Solve the nonlinear system using learned pp1 and pp2 values for x or Ci
    f = Model_equations(df, pp1, pp2)
    J1 = Jacobian(mtd="batchScalarJacobian_AD")#, func=f, settings={"dx":1e-2,"bounds":(10.0, 50.0)}) #mtd="batchScalarJacobian_AD"
    vG = tensorNewton(f,J1)#, mtd="rtsafe",lb=0,ub=1000)
    x0 = get_guess(c3c4_path_index, can_co2_ppress)
    x0.requires_grad_(True)
    x  = vG(x0)

    # Second: Compute different photosynthetic flux rates
    # ac    : Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
    # aj    : RuBP-limited gross photosynthesis (umol CO2/m**2/s)
    # ap    : product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
    # ai    : intermediate co - limited photosynthesis(umolCO2 / m ** 2 / s)
    # agross: co-limited gross leaf photosynthesis (umol CO2/m**2/s)
    # anet  : net leaf photosynthesis (umol CO2/m**2/s)
    ac = (vcmax * torch.clamp(x - co2_cpoint, min=0.0) / (
                x + mm_kco2 * (1.0 + can_o2_ppress / mm_ko2))) * c3c4_path_index \
         + vcmax * (1 - c3c4_path_index)

    aj = (je * torch.clamp(x - co2_cpoint, min=0.0) / (4.0 * x + 8.0 * co2_cpoint)) * c3c4_path_index + \
         (quant_eff * parsun_lsl * 4.6) * (1 - c3c4_path_index)

    ap = co2_rcurve_islope * torch.clamp(x, min=0.0) / can_press

    ai = 0.0 * c3c4_path_index + quadratic_min(theta_cj_c4, -(ac + aj), ac * aj) * (1 - c3c4_path_index)

    agross = quadratic_min(theta_cj_c3, -(ac + aj), ac * aj) * c3c4_path_index + \
             quadratic_min(theta_ip, - (ai + ap), ai * ap) * (1 - c3c4_path_index)

    anet   =  agross - lmr

    # Third:  correct anet for LAI > 0.0 and parsun_lsl <=0.0
    ### added them to give the correct values
    mask = LAI > 0.0
    mask = mask.type(torch.uint8)
    anet = mask * anet + (1 - mask) * 0.0

    mask = parsun_lsl <= 0.0
    mask = mask.type(torch.uint8)
    anet = mask * -lmr + (1-mask) * anet

    # Third: Stomatal Conductance computation
    a_gs = anet
    leaf_co2_ppress = torch.clamp(can_co2_ppress - h2o_co2_bl_diffuse_ratio / gb_mol * a_gs * can_press, min=1.e-06)
    #vpd = torch.clamp((veg_esat - ceair), min=50.0) * 0.001

    term_gsmol = h2o_co2_stoma_diffuse_ratio * anet / (leaf_co2_ppress / can_press)
    aquad = 1.0
    bquad = -(2.0 * (stomatal_intercept_btran_Med + term_gsmol) + (medlyn_slope * term_gsmol) ** 2 / (gb_mol * vpd))
    cquad = stomatal_intercept_btran_Med * stomatal_intercept_btran_Med + \
            (2.0 * stomatal_intercept_btran_Med + term_gsmol * (1.0 - medlyn_slope * medlyn_slope / vpd)) * term_gsmol

    gs_mol  = stomatal_intercept_btran_Med.clone()
    mask    = (anet < 0.0) | (bquad * bquad < 4.0 * aquad * cquad)  # I assume that complex roots appear only when anet approaches +-zero however this isnot working for the second condition due to nan values
    mask    = mask.type(torch.bool)
    gs_mol[torch.logical_not(mask)] = quadratic_max(aquad, bquad[torch.logical_not(mask)],cquad[torch.logical_not(mask)])

    # gs_mol2 = mask * stomatal_intercept_btran_Med + (1 - mask) * (h2o_co2_stoma_diffuse_ratio*(1 + medlyn_slope / torch.sqrt(vpd)) * \
    #           torch.clamp(anet,min = 0.0)/ leaf_co2_ppress*can_press + stomatal_intercept_btran_Med)

    return x, ac, aj, ap, leaf_co2_ppress, vpd, gs_mol
######################################################################################################################





