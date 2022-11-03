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
from solver import solve_direct

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
        self.phi_to_list = nn.ModuleList([Phi_to(2*self.latent_dimension + 2, self.latent_dimension) for i in range(self.k)])
        self.phi_from_list = nn.ModuleList([Phi_from(2*self.latent_dimension + 2, self.latent_dimension) for i in range(self.k)])
        self.phi_loop_list = nn.ModuleList([Loop(2*self.latent_dimension+1, self.latent_dimension) for i in range(self.k)])
        self.psy_list = nn.ModuleList([Psy(4*self.latent_dimension + 3, self.latent_dimension) for i in range(self.k)])
        self.decoder_list = nn.ModuleList([Decoder(self.latent_dimension, 2) for i in range(self.k)])

    def loss_function(self, F, x):
        loss = solve_direct(F, x)


        # loss_fn1 = nn.MSELoss()
        # loss_fn2 = torch.mean(torch.pow(torch.abs(torch.sub(F,y)), 0.5))
        # loss = loss_fn1(F, y)
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

        H['0'] = torch.zeros([batch.num_nodes, self.latent_dimension], dtype = torch.float, device = self.device)
        F['0'] = self.decoder_list[0](H['0'])# + self.U_init
        # set_trace()

        for update in range(self.k) :
            # set_trace()
            mess_to = self.phi_to_list[update](H[str(update)], batch.edge_index, batch.edge_attr)
            #print("Message_To size : ", mess_to.size())

            mess_from = self.phi_from_list[update](H[str(update)], batch.edge_index, batch.edge_attr)
            #print("Message_from size : ", mess_from.size())

            loop = self.phi_loop_list[update](H[str(update)], batch.edge_index, batch.edge_attr)
            #print("Message loop size :", loop.size())

            concat = torch.cat([H[str(update)], mess_to, mess_from, loop, batch.x], dim = 1)
            #concat = torch.cat([H[str(update)], mess_to, mess_from, loop, y], dim = 1)
            #print("Size concat : ", concat.size())

            correction = self.psy_list[update](concat)
            #print("Correction size : ", correction.size())
            #print(self.psy_list[update])

            H[str(update+1)] = H[str(update)] + self.alpha*correction
            #print("H+1 size : ", H[str(update+1)].size())

            if update >= self.k - 2:
                F[str(update+1)] = self.decoder_list[update](H[str(update+1)])
            #print("Size of U : ", U[str(update+1)].size())
            #print(self.decoder_list[update])

                loss[str(update+1)] = self.loss_function(F[str(update+1)], batch.x)

                if total_loss is None :
                    total_loss = loss[str(update+1)] * self.gamma**(self.k - update - 1)
                else :
                    total_loss += loss[str(update+1)] * self.gamma**(self.k - update - 1)

        #print(torch.mean((U[str(self.k-1)] - data.x)**2))

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

class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()

        self.MLP = nn.Sequential(   nn.Linear(in_size, in_size),
                                    nn.ReLU(),
                                    nn.Linear(in_size, out_size))
    def forward(self, x):

        return self.MLP(x)