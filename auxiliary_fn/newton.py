###
# THIS FILE INLCUDES ALL THE FUNCTIONS REQUIRED TO RUN THE NEWTON ITERATION SOLVER FOR THE NONLINEAR SYSTEM AND IT
# INCLUDES ALL THE EQUATIONS OF THE NONLINEAR SYSTEM IN THE LEAFLAYER PHOTOSYNTHESIS SUBROUTINE IN THE PHOTOSYNTHESIS
# MODULE IN FATES
###

import torch.nn as nn
from auxiliary_fn.Functions_FatesModel import *

def batchScalarJacobian_AD(x,y,graphed=True):
    # Description: extract the gradient dy/dx for scalar y (but with minibatch)
    # relying on the fact that the minibatch has nothing to do with each other!
    # y: [nb]; x: [nb, (?nx)]. For single-column x, get dydx as a vector [dy1/dx1,dy2/dx2,...,dyn/dxn]; For multiple-column x, there will be multi-column output
    # output: [nb, (nx?)]
    # if x is tuple/list, the return will also be tuple/list
    assert not (y.ndim>1 and y.shape[-1]>1), 'this function is only valid for batched scalar y outputs'
    gO = torch.ones_like(y,requires_grad=False) # donno why it does not do 2D output
    dydx=torch.autograd.grad(outputs=y, inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed)
    # calculate vjp. For the minibatch, we are taking advantage of the fact that the y at a site is unrelated to
    # the x at another site, so the matrix multiplication [1,1,...,1]*J reverts to extracting the diagonal of the Jacobian
    if isinstance(x,torch.Tensor):
        dydx=dydx[0] # it gives a tuple
    if not graphed:
        # during test, we detach the graph
        # without doing this, the following cannot be cleaned from memory between time steps as something use them outside
        if isinstance(dydx,torch.Tensor):
            dydx = dydx.detach()
        else:
            for dd in dydx:
                dd = dd.detach()
        y = y.detach()
        gO = gO.detach()
    return dydx


def batchJacobian_AD(x,y,graphed=True,doSqueeze=True):
    # Desription: extract the jacobian dy/dx for multi-column y output (and with minibatch)
    # compared to the scalar version above, this version will call grad() ny times and store outputs in a tensor matrix
    # y: [nb, ny]; x: [nb, nx]. x could also be a tuple or list of tensors.
    # permute and view your y to be of the above format.
    # AD jacobian is not free and may end up costing me time
    # output: Jacobian [nb, ny, nx] # will squeeze after the calculation
    # relying on the fact that the minibatch has nothing to do with each other!
    # if they do, i.e, they come from different time steps of a simulation, you need to put them in second dim in y!
    # view or reshape your x and y to be in this format if they are not!
    # pay attention, this operation could be expensive.
    ny = 1 if y.ndim==1 else y.shape[-1]

    # prepare the receptacle
    if isinstance(x,torch.Tensor):
        nx = 1 if x.ndim==1 else x.shape[-1]
        sizes = [y.shape[0],ny,nx]
        DYDX0 = torch.zeros(sizes,requires_grad=True)
        DYDX = DYDX0.clone() # avoid the "leaf variable cannot have in-place operation" issue
    elif isinstance(x,tuple) or isinstance(x,list):
        # prepare a list of tensors
        DYDX = list()
        for i in range(0,len(x)):
            nx = x[i].shape[-1]
            sizes = [y.shape[0],ny,nx]
            DYDX0 = torch.zeros(sizes,requires_grad=True)
            DYDX.append(DYDX0.clone()) # avoid the "leaf variable cannot have in-place operation" issue

    gO = torch.ones_like(y[:,0],requires_grad=False) # donno why it does not do 2D output
    for i in range(ny):
        dydx=torch.autograd.grad(outputs=y[:,i], inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed)
        if isinstance(x,torch.Tensor):
            DYDX[:,i,:]=dydx[0] # for some reason it gives me a tuple
            if doSqueeze:
                DYDX = DYDX.squeeze()
        elif isinstance(x,tuple) or isinstance(x,list):
            for j in range(len(x)):
                DYDX[j][:,i,:]=dydx[j] # for some reason it gives me a tuple
    #dydx2 = torch.autograd.grad(outputs=y[:,i], inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed, is_grads_batched=True)

    if not graphed:
        # during test, we may detach the graph
        # without doing this, the following cannot be cleaned from memory between time steps as something use them outside
        # however, if you are using the gradient during test, then graphed should be false.
        dydx = dydx.detach()
        DYDX0 = DYDX0.detach()
        DYDX = DYDX.detach()
        x = x.detach()
        y = y.detach()
        gO = gO.detach()
    return DYDX

def getDictJac(xDict, yDict, J, rt=1, xfs=("params","u0"), yfs=("yP",), ySel=([],), yPermuteDim=([],)):
    # Description: provides a succinct way of extracting Jacobian from a dictionary. Gives an interface to select indices along dimensions and permute.
    # JAC[i][j] is for yfs[i] and xfs[j]
    # NAMES describes these entries.
    # ySel is the tuple specifying selection (dim, tensor(index)). indexing the original data without changing number of dimensions.
    # permuteDim: if non-empty, will permute using this argument. permute happens after indexing
    X = list(); nx=1;
    for xf in xfs:
        dat = xDict[xf]
        X.append(dat) # it needs to append the whole thing, not a slice.
        if dat.ndim > 1:
            nx = nx * dat.shape[-1]

    JAC = list()
    NAMES = list()
    for yf,ys,pd in zip(yfs, ySel, yPermuteDim):
        d = yDict[yf]
        #d = selectDims(dat, (d0,d1,d2)).view([dat.shape[0], int(dat.nelement()/dat.shape[0])]) # essentially dat[:,d1,d2], but [] means select the entire axis. also safe with 1D/2D arrays.
        if len(ys)==2:
            d = torch.select(d,ys[0],ys[1])
        if len(pd)>0:
            d = torch.permute(d, pd)
        #d = torch.squeeze(d) # remove singleton dimension

        jac0 = J(X,d)
        JAC.append(jac0)

        names = list()
        for xf in xfs:
            names.append(f"d({yf}{ySel}{yPermuteDim})/d({xf})")
        NAMES.append(names)
    return JAC, NAMES


class Jacobian(nn.Module):
    # DescriptionL an wrapper for all the Jacobian options -- stores some options and provide a uniform interface
    # J=Jacobian((mtd="batchScalarJacobian_AD"),(func))
    # jac=J(x,y)
    # x can be a torch.Tensor, or a tuple/list of Tensors, in which case the return will be a tuple/list of Tensors
    def __init__(self, mtd=0, func=None, create_graph=True, settings={"dx":1e-2}):
        super(Jacobian, self).__init__()
        self.mtd = mtd
        self.func = func
        self.settings = settings
        self.create_graph = create_graph

    def forward(self, x, y):
        ny = 1 if y.ndim==1 else y.shape[-1]
        # adaptively select the right function
        if self.mtd == 0 or self.mtd == "batchScalarJacobian_AD": # we can also add or ny==1
            Jac = batchScalarJacobian_AD(x,y,graphed=self.create_graph)
        elif self.mtd == 1 or self.mtd == "batchJacobian_AD":
            Jac = batchJacobian_AD(x,y,graphed=self.create_graph, doSqueeze=False)
        return Jac

def rtnobnd(x0, G, J, settings, doPrint=False):
    # Description: solves the nonlinear problem with unbounded Newton iteration
    # may have poor global convergence. but if it works for your function it should be fast.
    x = x0.clone();
    nx = 1 if x.ndim==1 else x.shape[-1]
    iter=0; ftol=1e3; xtol=1e4

    while (iter<settings["maxiter"]) and (ftol>settings["ftol"]) and (xtol>settings["xtol"]):
        f = G(x)
        if torch.isnan(f).any():## for anet< 0 it gives nan stomatal conductance so we should break the loop
            print("True")
            break
        dfdx = J(x,f)
        if nx == 1:
            xnew = x - f/dfdx
        else:
            deltaX = torch.linalg.solve(dfdx, f)
            xnew   = x - deltaX
        ftol = f.abs().max()
        xtol = (xnew-x).abs().max()
        x    = xnew
        iter +=1
        if doPrint:
            print(f"iter={iter}, x= {float(x[0])}, dfdx= {float(dfdx[0])}, xtol= {xtol}, ftol= {ftol}") #x1164= {x[1164]}, f1164 = {f[1164]}, dfdx1164 = {dfdx[1164]}"
    return x


class tensorNewton(nn.Module):
    # Description: solve a nonlinear problem of G(x)=0 using Newton's iteration
    # x can be a vector of unknowns. [nb, nx] where nx is the number of unknowns and nb is the minibatch size
    # minibatch is for different sites, physical parameter sets, etc.
    # model(x) should produce the residual
    def __init__(self, G,J=Jacobian(), mtd=0, lb=None, ub=None, settings={"maxiter":10, "ftol":1e-6, "xtol":1e-6,"alpha":0.75}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(tensorNewton, self).__init__()
        self.G = G
        self.J = J
        self.mtd = mtd
        self.lb = lb
        self.ub = ub
        self.settings = settings


    def forward(self, x0):
        if self.mtd == 0 or self.mtd=='rtnobnd':
            return rtnobnd(x0,self.G,self.J,self.settings)
        else:
            assert self.mtd<=1, 'tensorNewton:: the nonlinear solver has not been implemented yet'


class Model_equations(torch.nn.Module):
    # Description: Inlcudes our nonlinear system for the leaflayer photosynthesis subroutine in
    # the photosynthesis module in FATES

    # Inputs                  :Forcing dataset which includes:
    # parsun_lsl              :PAR absorbed in sunlit leaves for this layer
    # c3c4_path_index         :Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
    # stomatal_intercept_btran:water-stressed minimum stomatal conductance (umol H2O/m**2/s)
    # mm_kco2                 :Michaelis-Menten constant for CO2 (Pa)
    # mm_ko2                  :Michaelis-Menten constant for O2 (Pa)
    # co2_cpoint              :CO2 compensation point (Pa)
    # gb_mol                  :leaf boundary layer conductance (umol H2O/m**2/s)
    # ceair                   :vapor pressure of air, constrained (Pa)
    # lmr                     :Leaf Maintenance Respiration  (umol CO2/m**2/s)
    # vcmax                   :maximum rate of carboxilation,
    # jmax                    :maximum electron transport rate,
    # co2_rcurve_islope       :initial slope of CO2 response curve (C4 plants)
    # medlyn_slope            :Slope for Medlyn stomatal conductance model method, the unit is KPa^0.5
    # can_o2_ppress           :Partial pressure of O2 NEAR the leaf surface (Pa)
    # can_press               :Air pressure NEAR the surface of the leaf (Pa)
    # leaf_co2_ppress         :Partial pressure of CO2 AT the leaf surface (Pa)
    # veg_esat                :saturation vapor pressure at veg_tempk (Pa)
    # qabs                    :PAR absorbed by PS II (umol photons/m**2/s)
    # LAI                     :Leaf Area Index
    # je                      :electron transport rate (umol electrons/m**2/s)
    def __init__(self, forcing, pp1, pp2):
        super(Model_equations, self).__init__()
        self.forcing = forcing
        self.pp1 = pp1
        self.pp2 = pp2

    # output = model.forward(input) # where input are for the known data points
    def forward(self, x):

        parsun_lsl, c3c4_path_index  , stomatal_intercept_btran_Med, mm_kco2, mm_ko2,\
        co2_cpoint, gb_mol           , ceair                       , lmr    , vcmax ,\
        jmax      , co2_rcurve_islope, medlyn_slope                , can_o2_ppress  ,\
        can_press , leaf_co2_ppress  , veg_esat                    , qabs           ,\
        je        , LAI,vpd          = pre_fates(self.forcing, self.pp1, self.pp2)

        x = x.clone()

        # First: Compute different photosynthetic flux rates
        # ac   : Rubisco-limited gross photosynthesis (umol CO2/m**2/s)
        # aj   : RuBP-limited gross photosynthesis (umol CO2/m**2/s)
        # ap   : product-limited (C3) or CO2-limited (C4) gross photosynthesis (umol CO2/m**2/s)
        # ai   : intermediate co - limited photosynthesis(umolCO2 / m ** 2 / s)
        # agross: co-limited gross leaf photosynthesis (umol CO2/m**2/s)
        # anet : net leaf photosynthesis (umol CO2/m**2/s)

        ac     =   (vcmax * torch.clamp(x - co2_cpoint, min = 0.0) / (x + mm_kco2 * (1.0 + can_o2_ppress / mm_ko2))) * c3c4_path_index \
                 + vcmax * (1 - c3c4_path_index)

        aj     =  (je * torch.clamp(x - co2_cpoint, min = 0.0) /(4.0 * x + 8.0 * co2_cpoint)) * c3c4_path_index + \
                  (quant_eff * parsun_lsl * 4.6) * (1 - c3c4_path_index)

        ap     =  co2_rcurve_islope * torch.clamp(x, min = 0.0) / can_press

        ai     =  0.0 * c3c4_path_index +  quadratic_min(theta_cj_c4, -(ac+aj) , ac * aj)  * (1 - c3c4_path_index)

        agross = quadratic_min(theta_cj_c3,  -(ac + aj) , ac * aj) * c3c4_path_index + \
                 quadratic_min(theta_ip, - (ai + ap), ai * ap) * (1 - c3c4_path_index)

        anet   =  agross - lmr

        # Second:  correct anet for LAI > 0.0 and parsun_lsl <=0.0
            ### added them to give the correct values even during solving the nonlinear system
        mask = LAI > 0.0
        mask = mask.type(torch.uint8)
        anet = mask * anet + (1 - mask) * 0.0
        mask = parsun_lsl <= 0.0
        mask = mask.type(torch.uint8)
        anet = mask * -lmr + (1-mask) * anet
            ### added them to give the correct values even during solving the nonlinear system

        # Third: Stomatal Conductance computation
        a_gs            = anet
        leaf_co2_ppress = torch.clamp(leaf_co2_ppress, min=1.e-06)
        can_co2_ppress  = torch.clamp(leaf_co2_ppress + h2o_co2_bl_diffuse_ratio / gb_mol * a_gs * can_press, min = 1.e-06)
        # vpd           = torch.clamp((veg_esat - ceair), min = 50.0) * 0.001

        term_gsmol  = h2o_co2_stoma_diffuse_ratio * anet / (leaf_co2_ppress / can_press)
        aquad       = 1.0
        bquad       = -(2.0 * (stomatal_intercept_btran_Med + term_gsmol) +(medlyn_slope * term_gsmol) ** 2 / (gb_mol * vpd))
        cquad       = stomatal_intercept_btran_Med * stomatal_intercept_btran_Med + \
                      (2.0*stomatal_intercept_btran_Med + term_gsmol * (1.0 - medlyn_slope * medlyn_slope/vpd))* term_gsmol

        gs_mol = stomatal_intercept_btran_Med.clone()
        mask   = (anet < 0.0) | (bquad * bquad < 4.0 * aquad * cquad)  # I assume that complex roots appear only when anet approaches +-zero however this isnot working for the second condition due to nan values
        mask   = mask.type(torch.bool)
        gs_mol[torch.logical_not(mask)] = quadratic_max(aquad, bquad[torch.logical_not(mask)], cquad[torch.logical_not(mask)])

        # Fourth: Compute (f(x) = 0 <===> f(Ci) = 0)
        f = x - (can_co2_ppress - anet * can_press * (h2o_co2_bl_diffuse_ratio * gs_mol + h2o_co2_stoma_diffuse_ratio
            * gb_mol) / (gb_mol * gs_mol))

        return f
def testJacobian():
    # Description: Function to test all the Jacobian options

    x = torch.tensor([[1.0,2.0],[3.0,0.6],[0.1,0.2]],requires_grad=True)
    y0 = torch.zeros([3,2],requires_grad=True)
    y = y0.clone() # has to do clone, it seems.
    k0 = torch.tensor([2.0,0.5,0.6,1.0],requires_grad=True).repeat([3,1])
    k = k0.clone()
    y[:,0]  = k[:,0]*x[:,0]+k[:,1]*x[:,1]+x[:,0]*x[:,1]
    y[:,1]  = k[:,2]*x[:,0]+k[:,3]*x[:,1]
    jac = batchJacobian_AD(x,y)
    print(jac.detach().numpy()) # expected:
    #tensor([[[4.0000, 1.5000],
    #     [0.6000, 1.0000]],
    #    [[2.6000, 3.5000],
    #     [0.6000, 1.0000]],
    #    [[2.2000, 0.6000],
    #     [0.6000, 1.0000]]], grad_fn=<CopySlices>)
    xDict=dict(); yDict=dict()
    xDict["u0"] = x
    xDict["params"] = k
    yDict["yP"] = y
    J0 = Jacobian(mtd="batchJacobian_AD")
    JAC, NAMES = getDictJac(xDict,yDict,J0,xfs=("params",'u0'),yfs=('yP',),ySel=([],),yPermuteDim=([],))

    f = Model_equations()

    J1 = Jacobian(mtd="batchScalarJacobian_AD")
    vG = tensorNewton(f,J1)
    x0 = torch.tensor([483002.0260870451],requires_grad=True) # FD can be run in eval mode
    x  = vG(x0)
    print(x)


    return jac





















