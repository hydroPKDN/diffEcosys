###
# THIS FILE INCLUDES ALL THE PARAMETERS AND FUNCTIONS REQUIRED TO BE TRANSLATED FROM THE PHOTOSYNTHESIS MODULE IN FATES TO RUN THE
# LEAFLAYERPHOTOSYNTHESIS SUBROUTINE
###

import torch


rgas_J_K_kmol         = 8314.4598 ;#universal gas constant [J/K/kmol]
t_water_freeze_k_1atm = 273.15    ;#freezing point of water at triple point (K)
umol_per_mol          = 1.0e6     ;#Conversion factor: micromoles per mole
mmol_per_mol          = 1000.0    ;#Conversion factor: milimoles per mole
umol_per_kmol         = 1.0e9     ;#Conversion factor: micromoles per kmole
rgas_J_K_mol          = 8.3144598 ;#universal gas constant [J/K/mol]
nearzero              = 1.0e-30   ;#FATES use this in place of logical comparisons between reals with zero, as the chances are their precisions are preventing perfect zero in comparison
c3_path_index         = 1         ;#Constants used to define C3 versus C4 photosynth pathways
itrue                 = 1         ;#Integer equivalent of true
medlyn_model          = 2         ;#Constants used to define conductance models
ballberry_model       = 1         ;#Constants used to define conductance models
molar_mass_water      = 18.0      ;#Molar mass of water (g/mol)

#params
mm_kc25_umol_per_mol = 404.9      ;#Michaelis-Menten constant for CO2 at 25C
mm_ko25_mmol_per_mol = 278.4      ;#Michaelis-Menten constant for O2 at 25C
co2_cpoint_umol_per_mol = 42.75   ;#CO2 compensation point at 25C
kcha                 = 79430.0    ;#activation energy for kc (J/mol)
koha                 = 36380.0    ;#activation energy for ko (J/mol)
cpha                 = 37830.0    ;#activation energy for cp (J/mol)

lmrha                = 46390.0    ;#activation energy for lmr (J/mol)
lmrhd                = 150650.0   ;#deactivation energy for lmr (J/mol)
lmrse                = 490.0      ;#entropy term for lmr (J/mol/K)
lmrc                 = 1.15912391 ;#scaling factor for high


quant_eff            = 0.05       ;#quantum efficiency, used only for C4 (mol CO2 / mol photons)
vcmaxha              = 65330      ;#activation energy for vcmax (J/mol)
vcmaxhd              = 149250     ;#deactivation energy for vcmax (J/mol)
vcmaxse              = 485        ;#entropy term for vcmax (J/mol/k)
jmaxha               = 43540      ;#activation energy for jmax (J/mol)
jmaxhd               = 152040     ;#deactivation energy for jmax (J/mol)
jmaxse               = 495        ;#entropy term for jmax (J/mol/k)
prec                 = 1e-8       ;#Avoid zeros to avoid Nans
stomatal_intercept   = [40000, 10000];#Unstressed minimum stomatal conductance 10000 for C3 and 40000 for C4 (umol m-2 s-1)

fnps                 = 0.15       ;#Fraction of light absorbed by non-photosynthetic pigments
theta_psii           = 0.7        ;#empirical curvature parameter for electron transport rate
theta_cj_c3          = 0.999      ;#empirical curvature parameters for ac, aj photosynthesis co-limitation, c3
theta_cj_c4          = 0.999      ;#empirical curvature parameters for ac, aj photosynthesis co-limitation, c4
theta_ip             = 0.999      ;#empirical curvature parameter for ap photosynthesis co-limitation
h2o_co2_bl_diffuse_ratio    = 1.4 ;#Ratio of H2O/CO2 gass diffusion in the leaf boundary layer (approximate)
h2o_co2_stoma_diffuse_ratio = 1.6 ;#Ratio of H2O/CO2 gas diffusion in stomatal airspace (approximate)
nscaler             =1.0          ;#leaf nitrogen scaling coefficient (assumed here as 1)
f_sun_lsl           =1.0          ;#fraction of sunlit leaves (assumed = 1)
init_a2l_co2_c3     = 0.7         ;#First guess on ratio between intercellular co2 and the atmosphere (C3)
init_a2l_co2_c4     = 0.4         ;#First guess on ratio between intercellular co2 and the atmosphere (C4)
rsmax0              = 2.0e8       ;#maximum stomatal resistance [s/m] (used across several procedures
molar_mass_ratio_vapdry = 0.622   ;#Approximate molar mass of water vapor to dry air

#vcmax25top         = [50, 65, 39, 62, 41, 58, 62, 54, 54, 78, 78, 78];
#k_lwp              = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
########################################################################################################
def quadratic_min(aquad,bquad,cquad):
    #Description: Solve for the minimum root of a quadratic equation
    #Copied from: FATES , cited there as : ! Solution from Press et al (1986) Numerical Recipes: The Art of Scientific
                                          #! Computing (Cambridge University Press, Cambridge), pp. 145.


    #aquad, bquad, cquad are the terms of a quadratic equation
    #r1 & r2 are the roots of the equation
    mask = bquad >= 0.0
    mask = mask.type(torch.uint8)
    q = -0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * mask +  \
        -0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * ( 1 - mask)

    r1 = q / aquad
    mask = q != 0.0
    mask = mask.type(torch.uint8)
    r2 = cquad / (q+prec) * mask + 1.e36 * ( 1 - mask)
    #RUN CHECK#
    #r2[torch.where(torch.isnan(r2))[0]] = 1.e36 * ( 1 - mask[torch.where(torch.isnan(r2))[0]])
    # r1 = (-0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)) )/ aquad
    # r2 = (-0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)))/ aquad


    return torch.min(r1,r2)

########################################################################################################
def quadratic_max(aquad,bquad,cquad):
    # Description: Solve for the maximum root of a quadratic equation
    # Copied from: FATES , cited there as : ! Solution from Press et al (1986) Numerical Recipes: The Art of Scientific
                                          # ! Computing (Cambridge University Press, Cambridge), pp. 145.


    # Inputs : aquad, bquad, cquad are the terms of a quadratic equation
    # outputs: r1 & r2 are the roots of the equation
    mask = bquad >= 0.0
    mask = mask.type(torch.uint8)
    q = -0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * mask +  \
        -0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * ( 1 - mask)

    r1 = q / aquad
    mask = q != 0.0
    mask = mask.type(torch.uint8)
    r2 = cquad / (q+prec) * mask + 1.e36 * ( 1 - mask)
    #RUN CHECK#
    #r2[torch.where(torch.isnan(r2))[0]] = 1.e36 * ( 1 - mask[torch.where(torch.isnan(r2))[0]])
    # r1 = (-0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)) )/ aquad
    # r2 = (-0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad)))/ aquad

    return torch.max(r1,r2)

########################################################################################################
def ft1_f(tl,ha):
    # DESCRIPTION:photosynthesis temperature response
    # Copied from: FATES

    # Inputs :
    # tl: leaf temperature in photosynthesis temperature function (K)
    # ha: activation energy in photosynthesis temperature function (J/mol)

    # outputs: parameter scaled to leaf temperature (tl)

    return torch.exp(ha/(rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm + 25)) *
                 (1.0-(t_water_freeze_k_1atm + 25.0)/tl))

########################################################################################################

def fth25_f(hd,se):
    # Description:scaling factor for photosynthesis temperature inhibition
    # Copied from: FATES

    # Inputs :
    # hd:deactivation energy in photosynthesis temp function (J/mol)
    # se:entropy term in photosynthesis temp function (J/mol/K)

    # outputs: parameter scaled to leaf temperature (tl)
    return 1.0 + torch.exp(torch.tensor(-hd + se * (t_water_freeze_k_1atm + 25.0)) /
                       (rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm+25.0)))


########################################################################################################

def fth_f(tl,hd,se,scaleFactor):
    # Description:photosynthesis temperature inhibition
    # Copied from: FATES
    # Inputs :
    # tl: leaf temperature in photosynthesis temperature function (K)
    # hd:deactivation energy in photosynthesis temp function (J/mol)
    # se:entropy term in photosynthesis temp function (J/mol/K)
    #scaleFactor  ! scaling factor for high temp inhibition (25 C = 1.0)


    return scaleFactor / (1.0 + torch.exp((-hd+se*tl) / (rgas_J_K_kmol * 1.0e-3 * tl)))

########################################################################################################
def QSat(tempk, RH):
    # Description:Computes saturation mixing ratio and the change in saturation
    # Copied from: CLM5.0

    # Parameters for derivative:water vapor
    a0 =  6.11213476;      a1 =  0.444007856     ; a2 =  0.143064234e-01 ; a3 =  0.264461437e-03
    a4 =  0.305903558e-05; a5 =  0.196237241e-07;  a6 =  0.892344772e-10 ; a7 = -0.373208410e-12
    a8 =  0.209339997e-15

    # Parameters For ice (temperature range -75C-0C)
    c0 =  6.11123516;      c1 =  0.503109514;     c2 =  0.188369801e-01; c3 =  0.420547422e-03
    c4 =  0.614396778e-05; c5 =  0.602780717e-07; c6 =  0.387940929e-09; c7 =  0.149436277e-11
    c8 =  0.262655803e-14;

    # Inputs:
    # tempk: temperature in kelvin
    # RH   : Relative humidty in fraction

    #outputs:
    # veg_esat: saturated vapor pressure at tempk (pa)
    # air_vpress: air vapor pressure (pa)
    td = torch.min(torch.tensor(100.0), torch.max(torch.tensor(-75.0), tempk - t_water_freeze_k_1atm))

    mask = td >= 0.0
    mask = mask.type(torch.uint8)
    veg_esat = (a0 + td*(a1 + td*(a2 + td*(a3 + td*(a4 + td*(a5 + td*(a6 + td*(a7 + td*a8)))))))) * mask + \
               (c0 + td*(c1 + td*(c2 + td*(c3 + td*(c4 + td*(c5 + td*(c6 + td*(c7 + td*c8)))))))) * (1 - mask)

    veg_esat = veg_esat * 100.0           # pa
    air_vpress = RH * veg_esat            #RH as fraction
    return veg_esat, air_vpress


########################################################################################################
def GetCanopyGasParameters(can_press, can_o2_partialpress, veg_tempk, air_tempk,
                            air_vpress, veg_esat, rb):

    # Description: calculates the specific Michaelis Menten Parameters (pa) for CO2 and O2, as well as
    # the CO2 compentation point.
    # Copied from: FATES

    # Inputs:
    # can_press          : Air pressure within the canopy (Pa)
    # can_o2_partialpress: Partial press of o2 in the canopy (Pa
    # veg_tempk          : The temperature of the vegetation (K)
    # air_tempk          : Temperature of canopy air (K)
    # air_vpress         : Vapor pressure of canopy air (Pa)
    # veg_esat           : Saturated vapor pressure at veg surf (Pa)
    # rb                 : Leaf Boundary layer resistance (s/m)

    # Outputs:
    # mm_kco2   :Michaelis-Menten constant for CO2 (Pa)
    # mm_ko2    :Michaelis-Menten constant for O2 (Pa)
    # co2_cpoint:CO2 compensation point (Pa)
    # cf        :conversion factor between molar form and velocity form of conductance and resistance: [umol/m3]
    # gb_mol    :leaf boundary layer conductance (umol H2O/m**2/s)
    # ceair     :vapor pressure of air, constrained (Pa)

    kc25 = (mm_kc25_umol_per_mol / umol_per_mol) * can_press
    ko25 = (mm_ko25_mmol_per_mol / mmol_per_mol) * can_press
    sco  = 0.5 * 0.209/ (co2_cpoint_umol_per_mol / umol_per_mol)
    cp25 = 0.5 * can_o2_partialpress / sco

    mask = (veg_tempk > 150.0) & (veg_tempk < 350.0)
    mask = mask.type(torch.uint8)
    mm_kco2    = (kc25 * ft1_f(veg_tempk, kcha)) * mask + 1.0 * (1 - mask)
    mm_ko2     = (ko25 * ft1_f(veg_tempk, koha)) * mask + 1.0 * (1 - mask)
    co2_cpoint = (cp25 * ft1_f(veg_tempk, cpha)) * mask + 1.0 * (1 - mask)

    cf = can_press / (rgas_J_K_kmol * air_tempk) * umol_per_kmol
    gb_mol = (1.0 / rb) * cf
    ceair = torch.min(torch.max(air_vpress, 0.05 * veg_esat), veg_esat)

    return mm_kco2, mm_ko2, co2_cpoint, cf, gb_mol, ceair


########################################################################################################
def lmr25top_ft_extract(c3c4_path_index, vcmax25top_ft):
    # Description: calculates the canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)
    # for C3 plants : lmr_25top_ft = 0.015 vcmax25top_ft
    # for C4 plants : lmr_25top_ft = 0.025 vcmax25top_ft

    # Inputs:
    # c3c4_path_index: index whether pft is C3 (index = 1) or C4 (index = 0)
    # vcmax25top_ft  : canopy top maximum rate of carboxylation at 25C for this pft (umol CO2/m**2/s)

    # Outputs:
    # lmr_25top_ft   : canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)

    mask = c3c4_path_index == c3_path_index
    mask = mask.type(torch.uint8)
    lmr_25top_ft = (0.015 * vcmax25top_ft) * mask + (0.025 * vcmax25top_ft) * (1 - mask)
    return lmr_25top_ft

########################################################################################################
def LeafLayerMaintenanceRespiration(lmr25top_ft, nscaler,veg_tempk, c3c4_path_index):
    # Description :  Base maintenance respiration rate for plant tissues maintresp_leaf_ryan1991_baserate
    # M. Ryan, 1991. Effects of climate change on plant respiration. It rescales the canopy top leaf maint resp
    # rate at 25C to the vegetation temperature (veg_tempk)

    # Inputs:
    # lmr_25top_ft   : canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)
    # nscaler        : leaf nitrogen scaling coefficient (assumed here as 1)
    # veg_tempk      : vegetation temperature
    # c3c4_path_index: index whether pft is C3 (index = 1) or C4 (index = 0)

    # Outputs:
    # lmr    : Leaf Maintenance Respiration  (umol CO2/m**2/s)

    lmr25 = lmr25top_ft * nscaler  ## nscaler =1
    mask  = c3c4_path_index == 1
    mask  = mask.type(torch.uint8)

    lmr = (lmr25 * ft1_f(veg_tempk, lmrha) * fth_f(veg_tempk, lmrhd, lmrse, lmrc)) * mask + \
          (lmr25 * 2.0 ** ((veg_tempk - (t_water_freeze_k_1atm + 25.0)) / 10.0) )  * (1 - mask)

    lmr = lmr * mask + (lmr / (1.0 + torch.exp(1.3 * (veg_tempk-(t_water_freeze_k_1atm+55.0))))) * (1 - mask)

    return lmr
########################################################################################################
def LeafLayerBiophysicalRates(parsun_lsl, vcmax25top_ft, jmax25top_ft, co2_rcurve_islope25top_ft, nscaler, veg_tempk
                              ,btran, c3c4_path_index):

    # Description: calculates the localized rates of several key photosynthesis rates.  By localized, we mean specific to the plant type and leaf layer,
    # which factors in leaf physiology, as well as environmental effects. This procedure should be called prior to iterative solvers, and should
    # have pre-calculated the reference rates for the pfts before this

    # Inputs:
    # parsun_lsl               :PAR absorbed in sunlit leaves for this layer
    # vcmax25top_ft            :canopy top maximum rate of carboxylation at 25C for this pft (umol CO2/m**2/s)
    # jmax25top_ft             :canopy top maximum electron transport rate at 25C for this pft (umol electrons/m**2/s)
    # co2_rcurve_islope25top_ft:initial slope of CO2 response curve (C4 plants) at 25C, canopy top, this pft
    # nscaler                  :leaf nitrogen scaling coefficient (assumed here as 1)
    # veg_tempk                :vegetation temperature
    # btran                    :transpiration wetness factor (0 to 1)
    # c3c4_path_index          :index whether pft is C3 (index = 1) or C4 (index = 0)

    # Outputs:
    # vcmax            :maximum rate of carboxylation (umol co2/m**2/s)
    # jmax             :maximum electron transport rate (umol electrons/m**2/s)
    # co2_rcurve_islope:initial slope of CO2 response curve (C4 plants)

   vcmaxc = fth25_f(vcmaxhd, vcmaxse)
   jmaxc = fth25_f(jmaxhd, jmaxse)

   vcmax25 = vcmax25top_ft * nscaler
   jmax25  = jmax25top_ft * nscaler
   co2_rcurve_islope25 = co2_rcurve_islope25top_ft * nscaler

   vcmax = vcmax25 * ft1_f(veg_tempk, vcmaxha) * fth_f(veg_tempk, vcmaxhd, vcmaxse, vcmaxc)
   jmax  = jmax25  * ft1_f(veg_tempk, jmaxha)  * fth_f(veg_tempk, jmaxhd, jmaxse, jmaxc)
   co2_rcurve_islope = co2_rcurve_islope25 * 2.0 ** ((veg_tempk - (t_water_freeze_k_1atm + 25.0)) / 10.0)

   mask = c3c4_path_index != 1
   mask = mask.type(torch.uint8)

   vcmax = vcmax * (1 - mask) + (vcmax25 * 2.0 ** ((veg_tempk - (t_water_freeze_k_1atm + 25.0)) / 10.0)) * mask
   vcmax = vcmax * (1 - mask) + (vcmax / (1.0 + torch.exp(0.2 * ((t_water_freeze_k_1atm+15.0) - veg_tempk)))) * mask
   vcmax = vcmax * (1 - mask) + (vcmax / (1.0 + torch.exp(0.3 * (veg_tempk-(t_water_freeze_k_1atm+40.0))))) * mask

   mask = (parsun_lsl <= 0.0)
   mask = mask.type(torch.uint8)

   vcmax = 0.0 * mask + vcmax * (1 - mask)
   jmax  = 0.0 * mask + jmax  * (1 - mask)
   co2_rcurve_islope = 0.0 * mask + co2_rcurve_islope * (1 - mask)

   vcmax = vcmax * btran



   return vcmax, jmax, co2_rcurve_islope
########################################################################################################
def get_data(df_input):
    # Description: This function extracts all the data variables required to run the whole model
    # Inputs     : The whole dataset


    # Outputs    :
    # can_press               :Air pressure NEAR the surface of the leaf (Pa)
    # can_o2_ppress           :Partial pressure of O2 NEAR the leaf surface (Pa)
    # leaf_co2_ppress         :Partial pressure of CO2 AT the leaf surface (Pa)
    # veg_tempk               :Leaf temperature     [K]
    # parsun_lsl              :PAR absorbed in sunlit leaves for this layer
    # veg_esat                :saturation vapor pressure at veg_tempk (Pa)
    # c3c4_path_index         :Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
    # mm_kco2                 :Michaelis-Menten constant for CO2 (Pa)
    # mm_ko2                  :Michaelis-Menten constant for O2 (Pa)
    # co2_cpoint              :CO2 compensation point (Pa)
    # cf                      :conversion factor between molar form and velocity form of conductance and resistance: [umol/m3]
    # gb_mol                  :leaf boundary layer conductance (umol H2O/m**2/s)
    # ceair                   :vapor pressure of air, constrained (Pa)
    # stomatal_intercept_btran:water-stressed minimum stomatal conductance (umol H2O/m**2/s)
    # medlyn_slope            :Slope for Medlyn stomatal conductance model method, the unit is KPa^0.5
    # qabs                    :PAR absorbed by PS II (umol photons/m**2/s)
    # LAI                     :Leaf Area Index

    can_press       = torch.FloatTensor(df_input.Patm.values * 1000)
    can_o2_ppress   = 0.209 * can_press
    leaf_co2_ppress = torch.FloatTensor(df_input.CO2S.values * df_input.Patm.values /1000)
    veg_tempk       = torch.FloatTensor(df_input.Tleaf.values + 273.15)
    air_tempk       = torch.FloatTensor(df_input.Tair.values + 273.15)
    RH              = torch.FloatTensor(df_input.RH.values)
    vpd             = torch.FloatTensor(df_input.VPD.values)
    parsun_lsl      = torch.FloatTensor(df_input.PARin.values /4.6)
    veg_esat        = torch.FloatTensor(QSat(veg_tempk, RH)[0])
    air_vpress      = torch.FloatTensor(QSat(veg_tempk, RH)[1])
    BLCond          = torch.FloatTensor(df_input.BLCond.values)
    rb              = ((can_press * umol_per_kmol) / (rgas_J_K_kmol * air_tempk)) /(BLCond * 10**6)
    c3c4_pathway    = df_input.Pathway.values
    c3c4_path_index = torch.IntTensor([1 if c3c4_pathway[i] == "C3" else 0 for i in range(len(c3c4_pathway))])
    LAI             = torch.FloatTensor(df_input.LAI.values)

    # GetCanopyGasParameters
    mm_kco2, mm_ko2, co2_cpoint, cf, gb_mol, ceair = GetCanopyGasParameters(can_press , can_o2_ppress,
                                                                            veg_tempk , air_tempk    ,
                                                                            air_vpress,veg_esat      ,rb)

    #compute qabs
    qabs           = parsun_lsl * 0.5 * (1.0 - fnps) * 4.6

    # Get Stomatal Conductance parameters
    stomatal_intercept_btran = torch.FloatTensor([stomatal_intercept[i] for i in c3c4_path_index])
    medlyn_slope             = torch.FloatTensor(df_input.medslope.values)

    return (    can_press               , can_o2_ppress  , leaf_co2_ppress, veg_tempk, parsun_lsl   ,\
                veg_esat                , c3c4_path_index, mm_kco2        , mm_ko2   ,co2_cpoint    ,\
                cf                      , gb_mol         , ceair          , stomatal_intercept_btran,\
                medlyn_slope            , qabs           , LAI            , vpd)

def pre_fates(forcing,vcmax25top_ft, btran):

    # Description: This function computes takes vcmax25 and btran as inputs and output all forcings required
    # for the differentiable model

    # Inputs : Outputs from get_data(df_input)

    # Outputs:
    # parsun_lsl              :PAR absorbed in sunlit leaves for this layer
    # c3c4_path_index         :Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
    # stomatal_intercept_btran:water-stressed minimum stomatal conductance (umol H2O/m**2/s)
    # mm_kco2                 :Michaelis-Menten constant for CO2 (Pa)
    # mm_ko2                  :Michaelis-Menten constant for O2 (Pa)
    # co2_cpoint              :CO2 compensation point (Pa)
    # gb_mol                  :leaf boundary layer conductance (umol H2O/m**2/s)
    # ceair                   :vapor pressure of air, constrained (Pa)
    # lmr                     :Leaf Maintenance Respiration  (umol CO2/m**2/s)
    # vcmax                   :maximum rate of carboxylation (umol co2/m**2/s)
    # jmax                    :maximum electron transport rate (umol electrons/m**2/s)
    # co2_rcurve_islope       :initial slope of CO2 response curve (C4 plants)
    # medlyn_slope            :Slope for Medlyn stomatal conductance model method, the unit is KPa^0.5
    # can_o2_ppress           :Partial pressure of O2 NEAR the leaf surface (Pa)
    # can_press               :Air pressure NEAR the surface of the leaf (Pa)
    # leaf_co2_ppress         :Partial pressure of CO2 AT the leaf surface (Pa)
    # veg_esat                :saturation vapor pressure at veg_tempk (Pa)
    # qabs                    :PAR absorbed by PS II (umol photons/m**2/s)
    # LAI                     :Leaf Area Index
    # je                      :electron transport rate (umol electrons/m**2/s)

    can_press               , can_o2_ppress  , leaf_co2_ppress, veg_tempk, parsun_lsl   ,\
    veg_esat                , c3c4_path_index, mm_kco2        , mm_ko2   ,co2_cpoint    ,\
    cf                      , gb_mol         , ceair          , stomatal_intercept_btran,\
    medlyn_slope            , qabs           , LAI            , vpd = forcing

    jmax25top_ft                   = 1.67  * vcmax25top_ft
    co2_rcurve_islope25top_ft      = 20000 * vcmax25top_ft
    lmr25top_ft                    = lmr25top_ft_extract(c3c4_path_index, vcmax25top_ft)

    lmr                            = LeafLayerMaintenanceRespiration(lmr25top_ft, nscaler,veg_tempk, c3c4_path_index)

    vcmax, jmax ,co2_rcurve_islope =  LeafLayerBiophysicalRates(parsun_lsl, vcmax25top_ft, jmax25top_ft,
                                      co2_rcurve_islope25top_ft, nscaler,veg_tempk, btran, c3c4_path_index)

    stomatal_intercept_btran_Med   = torch.max(cf/rsmax0, stomatal_intercept_btran * btran)
    je                             = quadratic_min(theta_psii, - (qabs + jmax),  qabs * jmax)

    return(parsun_lsl, c3c4_path_index  , stomatal_intercept_btran_Med,mm_kco2, mm_ko2,
           co2_cpoint, gb_mol           , ceair                       , lmr   , vcmax ,
           jmax      , co2_rcurve_islope, medlyn_slope                , can_o2_ppress ,
           can_press , leaf_co2_ppress  , veg_esat                    , qabs          ,
           je        , LAI              , vpd)









