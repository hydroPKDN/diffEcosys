###
# THIS FILE INCLUDES DIFFERENT STRUCTURES AND CONFIGURATION OF NEURAL NETWORKS MODELS INCLUDING THE ONES USED IN THIS
# PROJECT
###

import torch
import torch.nn as nn

#####################################################################################################################
def embszs_define(cols, df):
    # Description: Define the embedding size used to convert categorical inputs into quantitative values
    # Inputs:
    # cols  : The categorical columns in your dataset
    # df    : The dataframe including your data

    # Outputs:
    # emb_szs: The embedding size as a tuple
    #cat_szs = [len(df[col].cat.categories) for col in cols]
    cat_szs = [df[col].nunique() for col in cols]
    emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
    return emb_szs
#####################################################################################################################
class Network(nn.Module):
    # Network of three layers : Input, hidden, and output layer
    # Sigmoid activation functions are used for both hidden and output layers
    def __init__(self, In, nh, out):
        super().__init__()

        self.hidden = nn.Linear(In, nh)
        self.output = nn.Linear(nh, out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

#####################################################################################################################
class Network_sig(nn.Module):
    # A more than the previous versions that can include different numbers of layers as indicated by the user.

    # Sigmoid activation functions are used for the hidden layers and NONE for the output layer
    def __init__(self, n_cont, out_sz, layers, p = 0.4):
        super().__init__()
        self.emb_drop = nn.Dropout(p)
        layerlist = []
        n_in =  n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.Sigmoid())
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        layerlist.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cont):
        x = torch.cat([x_cont],1)
        x = self.layers(x)
        return x


#####################################################################################################################
class TabularModel_sig(nn.Module):
    # A more flexible than the previous versions that can include different numbers of layers as indicated by the user.
    # It can also digest categorical and continuous inputs simultaneously through the embedding layer

    # Sigmoid activation functions are used for the hidden layers and the output layer
    def __init__(self, emb_szs, n_cont, out_sz, layers, p = 0.4):
        super().__init__()

        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        #self.bn_cont = nn.BatchNorm1d(n_cont)
        layerlist = []
        n_embs = sum([nf for ni,nf in emb_szs])
        n_in = n_embs + n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.Sigmoid())
            #layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        layerlist.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))

        x = torch.cat(embeddings,1)
        x = self.emb_drop(x)
        #x_cont = self.bn_cont(x_cont)
        x = torch.cat([x,x_cont],1)
        x = self.layers(x)
        return x
#####################################################################################################################
class TabularModel_relu(nn.Module):
    # A more flexible than the previous versions that can include different numbers of layers as indicated by the user.
    # It can also digest categorical and continuous inputs simultaneously through the embedding layer

    # RELU activation functions are used for the hidden layers and the output layer
    def __init__(self, emb_szs, n_cont, out_sz, layers, p = 0.4):
        super().__init__()

        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        layerlist = []
        n_embs = sum([nf for ni,nf in emb_szs])
        n_in = n_embs + n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.Dropout(p))
            n_in = i

        layerlist.append(nn.Linear(layers[-1], out_sz))
        layerlist.append(nn.ReLU())
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))

        x = torch.cat(embeddings,1)
        x = self.emb_drop(x)
        x = torch.cat([x,x_cont],1)
        x = self.layers(x)
        return x
#####################################################################################################################




