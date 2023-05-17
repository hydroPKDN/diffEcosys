###
# THIS FILE INCLUDES ALL THE FUNCTIONS REQUIRED TO TRAIN DIFFERENT MODEL FORMULATIONS IN THIS PROJECT
# 1. Train V_only (V + B_def)
# 2. Train B_only (V_def + B)
# 3. Train V_B (V + B)
###

import torch
from auxiliary_fn.predict  import predict
from auxiliary_fn.NN_input import get_btran

########################################################################################################################
def Train_Vonly(epochs , Vcat, target, vcmax_max, vcmax_min, NN_v, forcing, btran, criterion):

    # Description: A function to train a model to learn Vcmax25 only
    # Inputs:
    # epochs   : Number of epochs
    # V_input  : Input used for Vcmax25 neural network
    # target   : The target variable (in our case the observation for the net photosynthetic rates)
    # vcmax_max: a maximum limit for Vcmax25 to rescale the output of the neural network
    # vcmax_min: a minimum limit for Vcmax25 to rescale the output of the neural network
    # NN_v     : The model to be trained for learning Vcmax25
    # forcing  : all the forcing variables required to predict the net photosynthetic rates using predict function
    # btran    : soil water stress factor
    # criterion: The loss function used to train the model
    optimizer=torch.optim.Adam(NN_v.parameters(), lr=0.045)
    for i in range(epochs):
        vcmax25 = NN_v(Vcat)
        vcmax25 = vcmax25 * (vcmax_max - vcmax_min) + vcmax_min
        vcmax25 = vcmax25.reshape(-1)

        simulated, ftol = predict(forcing, vcmax25, btran)
        loss = torch.sqrt(criterion(simulated, target))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch = {}, loss = {:5.10f}, ftol = {:e}".format(i, loss, ftol.abs().max()))

    return NN_v
########################################################################################################################
def Train_Bonly(epochs , Bcat, Bcont,btran_params, target, vcmax25, NN_b, forcing, criterion, nly):
    # Description: A function to train a model to learn the parameter B only
    # Inputs:
    # epochs   : Number of epochs
    # Bcat     : categorical input used for B neural network
    # Bcont    : continuous input used for B neural network
    # btran_params: parameters used in the calculation of btran along with B
    # target   : The target variable (in our case the observation for the net photosynthetic rates)
    # vcmax25  : The maximum carboxylation rate at 25 C
    # NN_b     : The model to be trained for learning the parameter B
    # forcing  : all the forcing variables required to predict the net photosynthetic rates using predict function
    # criterion: The loss function used to train the model
    # nly      : Number of soil layers considered
    # dev      : The working device whether 'cpu' of 'Cuda'

    optimizer=torch.optim.Adam(NN_b.parameters(), lr=0.045)
    for i in range(epochs):

        B = NN_b(Bcat, Bcont).reshape(-1)
        btran = get_btran(btran_params, B, nly)

        simulated, ftol = predict(forcing, vcmax25, btran)

        loss = torch.sqrt(criterion(simulated, target))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("Epoch = {}, loss = {:5.10f}, ftol = {:e}".format(i, loss, ftol.abs().max()))

    return NN_b
########################################################################################################################
def TrainV_B(epochs ,Vcat, Bcat, Bcont,btran_params, target, NN_v, NN_b, forcing, criterion, nly,
             vcmax_max, vcmax_min):
    # Description: A function to train a model to learn the parameter B and Vcmax25

    # Inputs:
    # epochs   : Number of epochs
    # Bcat     : categorical input used for B neural network
    # Bcont    : continuous input used for B neural network
    # btran_params: parameters used in the calculation of btran along with B
    # target   : The target variable (in our case the observation for the net photosynthetic rates)
    # NN_v     : The model to be trained for learning Vcmax25
    # NN_b     : The model to be trained for learning the parameter B
    # forcing  : all the forcing variables required to predict the net photosynthetic rates using predict function
    # criterion: The loss function used to train the model
    # nly      : Number of soil layers considered
    # dev      : The working device whether 'cpu' of 'Cuda'
    # vcmax_max: a maximum limit for Vcmax25 to rescale the output of the neural network
    # vcmax_min: a minimum limit for Vcmax25 to rescale the output of the neural network

    params = list(NN_v.parameters()) + list(NN_b.parameters())
    optimizer=torch.optim.Adam(params, lr=0.045)
    for i in range(epochs):

        vcmax25 = NN_v(Vcat)
        vcmax25 = vcmax25 * (vcmax_max - vcmax_min) + vcmax_min
        vcmax25 = vcmax25.reshape(-1)

        if Bcat != None:
            B = NN_b(Bcat, Bcont).reshape(-1)
        else:
            B = NN_b(Bcont).reshape(-1)
        btran = get_btran(btran_params, B, nly)

        simulated, ftol = predict(forcing, vcmax25, btran)
        loss            = torch.sqrt(criterion(simulated, target))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("Epoch = {}, loss = {:5.10f}, ftol = {:e}".format(i, loss, ftol.abs().max()))

    return NN_v, NN_b
########################################################################################################################
# def TrainV_B_subhourly(epochs ,Vcat, Bcat, Bcont,btran_params, target, NN_v, NN_b, forcing, criterion, nly,
#              vcmax_max, vcmax_min, Vcont, target_norm):
#     # Description: A function to train a model to learn the parameter B and Vcmax25
#
#     # Inputs:
#     # epochs   : Number of epochs
#     # Bcat     : categorical input used for B neural network
#     # Bcont    : continuous input used for B neural network
#     # btran_params: parameters used in the calculation of btran along with B
#     # target   : The target variable (in our case the observation for the net photosynthetic rates)
#     # NN_v     : The model to be trained for learning Vcmax25
#     # NN_b     : The model to be trained for learning the parameter B
#     # forcing  : all the forcing variables required to predict the net photosynthetic rates using predict function
#     # criterion: The loss function used to train the model
#     # nly      : Number of soil layers considered
#     # dev      : The working device whether 'cpu' of 'Cuda'
#     # vcmax_max: a maximum limit for Vcmax25 to rescale the output of the neural network
#     # vcmax_min: a minimum limit for Vcmax25 to rescale the output of the neural network
#
#     params = list(NN_v.parameters()) + list(NN_b.parameters())
#     optimizer=torch.optim.Adam(params, lr=0.045)
#     #r, SMC, epsi_c, epsi_o = get_btran_params(df, dev, nly)
#     for i in range(epochs):
#
#         vcmax25 = NN_v(Vcat,Vcont)
#         vcmax25 = vcmax25 * (vcmax_max - vcmax_min) + vcmax_min
#         vcmax25 = vcmax25.reshape(-1)
#
#         if Bcat != None:
#             B = NN_b(Bcat, Bcont).reshape(-1)
#         else:
#             B = NN_b(Bcont).reshape(-1)
#         btran = get_btran(btran_params, B, nly)
#
#         simulated, ftol   = predict(forcing, vcmax25, btran)
#         simulated         = (simulated - target.mean())/target.std()
#         #l2_regularization = torch.zeros_like(ftol[0])   ;l2_lambda = 5e-4  # 0.006 for normalized
#         #for param in params:
#         #    l2_regularization += torch.norm(param, 2) ** 2
#         loss            = torch.sqrt(criterion(simulated, target_norm)) #+ l2_lambda * l2_regularization
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         print("Epoch = {}, loss = {:5.10f}, ftol = {:e}".format(i, loss, ftol.abs().max()))
#
#     return NN_v, NN_b
########################################################################################################################
def Train_NN(epochs, criterion, optimizer, model, cat_input, cont_input, target):
    # Description: A function used to train a neural network model
    # Inputs:
    # epochs   : Number of epochs
    # criterion: The loss function used to train the model
    # optimizer: The chosen optimizer (ADAM, SGD....)
    # model    : The model to be trained for learning
    # cat_input    : The categorical inputs for the trained model
    # cont_input   : The continuous inputs for the trained model
    # target   : The target variable (in our case the observation for the net photosynthetic rates)

    # Outputs:
    # model : The trained model
    # Losses: The losses array
    Losses = []

    for i in range(epochs):
        optimizer.zero_grad()
        if len(cat_input) > 0:
            simulated = model(cat_input, cont_input)
        else:
            simulated = model(cont_input)
        #l2_regularization =  torch.tensor(0.0, device = gpuid); l2_lambda = 0.0005
        #for param in params:
             #l2_regularization += torch.norm(param, 2)**2
        loss = torch.sqrt(criterion(simulated.reshape(-1), target)) #l2_lambda * l2_regularization
        loss.backward()
        optimizer.step()
        Losses.append(loss.item())
        print("Epoch = {}, loss = {}".format(i, loss))

    return model, Losses
########################################################################################################################
def Test_NN(trained_model,  cat_input, cont_input):
    # Description: A function used to test your trained model
    # Inputs:
    # trained_model: The model trained on the training dataset
    # cat_input    : The categorical inputs for the trained model
    # cont_input   : The continuous inputs for the trained model

    trained_model.eval()
    with torch.no_grad():
        if len(cat_input) > 0:
            pred = trained_model(cat_input, cont_input).to('cpu').detach().reshape(-1).numpy()
        else:
            pred = trained_model(cont_input).to('cpu').detach().reshape(-1).numpy()

    return pred
########################################################################################################################
