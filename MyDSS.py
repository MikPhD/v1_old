import numpy as np
import torch
import time
import torchvision
import torch.nn as nn
from torch_geometric import utils, data
from torch_geometric.nn import MessagePassing
from MyTrain import Train_DSS
from pdb import set_trace
import sys
import os


class MyOwnDSSNet(nn.Module):

    def __init__(self, latent_dimension, k, gamma, alpha, device):
        super(MyOwnDSSNet, self).__init__()

        #Hyperparameters
        self.latent_dimension = latent_dimension
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        #Neural network
        self.phi_to = Phi_to(2*self.latent_dimension + 2, self.latent_dimension)
        self.phi_from = Phi_from(2*self.latent_dimension + 2, self.latent_dimension)
        self.phi_loop = Loop(2*self.latent_dimension+1, self.latent_dimension)
        self.psy = Psy(4*self.latent_dimension + 3, self.latent_dimension)
        self.recurrent = Recurrent(4*self.latent_dimension+3, self.latent_dimension)
        self.decoder = Decoder(self.latent_dimension, 2)


    def loss_function(self, F, y):
        loss_fn = nn.MSELoss()
        loss = loss_fn(F, y)
        # loss = torch.norm(F - y)/torch.norm(y)
        # loss = (F - y)
        return loss

    def forward(self, batch):

        #Initialisation
        H = {}
        F = {}
        loss = {}
        total_loss = None

        self.F_init = batch.x*0

        H = torch.zeros([batch.num_nodes, self.latent_dimension], dtype = torch.float, device = self.device)
        # H_tot = torch.zeros([self.k, batch.num_nodes, self.latent_dimension], dtype=torch.float, device=self.device)

        F = self.decoder(H)# + self.U_init
        # set_trace()

        for update in range(self.k) :
            # set_trace()
            mess_to = self.phi_to(H, batch.edge_index, batch.edge_attr)
            #print("Message_To size : ", mess_to.size())

            mess_from = self.phi_from(H, batch.edge_index, batch.edge_attr)
            #print("Message_from size : ", mess_from.size())

            loop = self.phi_loop(H, batch.edge_index, batch.edge_attr)
            #print("Message loop size :", loop.size())

            concat = torch.cat([H, mess_to, mess_from, loop, batch.x], dim = 1)
            #concat = torch.cat([H[str(update)], mess_to, mess_from, loop, y], dim = 1)
            #print("Size concat : ", concat.size())

            # elaborate = self.psy(concat)
            #
            # H = H + self.alpha * elaborate

            # H_tot[update, : , :] = H

            new_embedded, last_hidden = self.recurrent(concat)
            new_embedded_elaborate = torch.squeeze(new_embedded, 1)

            H = new_embedded_elaborate * self.alpha + H

            F = self.decoder(H)

            loss[str(update+1)] = self.loss_function(F, batch.y)
            #
            if total_loss is None :
                total_loss = loss[str(update+1)] * self.gamma**(self.k - update - 1)
            else :
                total_loss += loss[str(update+1)] * self.gamma**(self.k - update - 1)

            #
            # correction, _ = self.recurrent(elaborate)
            # correction = torch.squeeze(correction, 0)
            ##correction = torch.reshape(correction, (correction.shape[0], correction.shape[1]))
            #
            #print("Correction size : ", correction.size())
            #print(self.psy_list[update])


            #print("H+1 size : ", H[str(update+1)].size())

            # F = self.decoder(H)
            # #print("Size of U : ", U[str(update+1)].size())
            # #print(self.decoder_list[update])
            #
            # loss[str(update+1)] = self.loss_function(F, batch.y)
            # #
            # if total_loss is None :
            #     total_loss = loss[str(update+1)] * self.gamma**(self.k - update - 1)
            # else :
            #     total_loss += loss[str(update+1)] * self.gamma**(self.k - update - 1)

            # if update + 1 == self.k:
            #     F = self.decoder(H)
            #     loss[str(update + 1)] = self.loss_function(F, batch.y)
            #     total_loss = loss[str(update + 1)]

        return F, total_loss, loss

#######################################################################################################################################################
####################################################### NEURAL NETWORKS ###############################################################################
#######################################################################################################################################################

class Phi_to(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Phi_to, self).__init__(aggr='mean', flow = 'source_to_target')
        self.MLP = nn.Sequential(   nn.Linear(in_channels, out_channels),
                                    nn.ReLU(),
                                    nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):

        edge_index, edge_attr = utils.dropout_adj(edge_index, edge_attr, p=0.2)

        edge_index, edge_attr = utils.remove_self_loops(edge_index, edge_attr)

        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)

        return self.MLP(tmp)

class Phi_from(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Phi_from, self).__init__(aggr='mean', flow = "target_to_source")
        self.MLP = nn.Sequential(   nn.Linear(in_channels, out_channels),
                                    nn.ReLU(),
                                    nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = utils.dropout_adj(edge_index, edge_attr, p=0.2)

        edge_index, edge_attr = utils.remove_self_loops(edge_index, edge_attr)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)

        return self.MLP(tmp)

class Loop(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(Loop, self).__init__()
        self.MLP = nn.Sequential(   nn.Linear(in_channels, out_channels),
                                    nn.ReLU(),
                                    nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = utils.dropout_adj(edge_index, edge_attr, p=0.2)

        edge_index, edge_attr = utils.add_self_loops(edge_index, edge_attr[:,0], num_nodes = x.size(0))

        adj = utils.to_scipy_sparse_matrix(edge_index, edge_attr)
        loop = 1 - torch.tensor(adj.diagonal().reshape(-1,1), dtype = torch.float)
        loop = loop.to(self.device)
        tmp = torch.cat([x, x, loop], dim = 1)

        return self.MLP(tmp)

class Psy(nn.Module):
    def __init__(self, in_size, out_size):
        super(Psy, self).__init__()

        self.MLP = nn.Sequential(   nn.Linear(in_size, out_size),
                                    nn.ReLU(),
                                    nn.Linear(out_size, out_size))
    def forward(self, x): #dimensione H + fi + fi + loop +B
        return self.MLP(x)

class Recurrent(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Recurrent, self).__init__()

        self.GRU = nn.GRU(input_size=in_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x): #dimensione H + fi + fi + loop +B
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        return self.GRU(x)


class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()

        self.MLP = nn.Sequential(   nn.Linear(in_size, in_size),
                                    nn.ReLU(),
                                    nn.Linear(in_size, out_size))
    def forward(self, x):

        return self.MLP(x)