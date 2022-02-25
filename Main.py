from Mydataset import MyOwnDataset
from MyDSS import MyOwnDSSNet
from MyTrain import Train_DSS
from MyCreateData import CreateData
import argparse
import sys
import torch
import os
import shutil
from torch_geometric.data import DataListLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from pdb import set_trace
import json

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', help='epoch number', type=int, default=10)
parser.add_argument('-r', '--restart', type=eval, default=False, choices=[True, False], help='Restart training option')
parser.add_argument('-tcase', '--traincase', help='train cases', nargs="+", default=['40'])
parser.add_argument('-vcase', '--valcase', help='validation cases', nargs="+", default=['40'])
parser.add_argument('-n_out', '--n_output', help='output each n_out epoch', type=int, default=10)

args = parser.parse_args()

n_epoch = args.n_epoch
restart = args.restart
train_cases = args.traincase
val_cases = args.valcase
n_output = args.n_output

# train_cases = ['40','50','60','70','80','90','100','120','130','140','150']
# train_cases = ['40']
# val_cases = ['40']
# test_cases = ['110']

## Copy Mesh file in Results - needed for plot ##
for i in val_cases:
    src = os.path.join("../Dataset", i, "Mesh.h5")
    dst = "./Results/Mesh.h5"
    shutil.copyfile(src, dst)

## Setting blank for new execution ##
if not restart:
    if os.path.exists("./dataset/processed/data_val.pt"):
        os.remove("./dataset/processed/data_val.pt")
    if os.path.exists("./dataset/processed/data_train.pt"):
        os.remove("./dataset/processed/data_train.pt")
    if os.path.exists("./dataset/processed/pre_filter.pt"):
        os.remove("./dataset/processed/pre_filter.pt")
    if os.path.exists("./dataset/processed/pre_transform.pt"):
        os.remove("./dataset/processed/pre_transform.pt")
    if os.path.exists("./Model/best_model.pt"):
        os.remove("./Model/best_model.pt")
    if os.path.exists("./Model/best_model_normal_final.pt"):
        os.remove("./Model/best_model_normal_final.pt")

print("#################### DATA ADAPTING FOR GNN #######################")
# createdata = CreateData()
# createdata.transform(train_cases, 'train')
# createdata.transform(val_cases, 'val')

k_list=[30, 50, 70]
latent_dimension=[5,10,20]
gamma_list=[0.9]
alpha_list=[1e-3]
lr_list=[0.01]


#check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on : ', device)

for k in k_list:
    for latent_dimension in latent_dimension:
        for gamma in gamma_list:
            for alpha in alpha_list:
                for lr in lr_list:

                    set_name = str(k) +'-'+ str(latent_dimension).replace(".","") +'-'+ str(alpha).replace(".","") +'-'+ str(lr).replace(".","")
                    print("PARAMETER SET: k:{}, gamma:{}, alpha:{}, lr:{}".format(str(k), str(latent_dimension), str(alpha), str(lr)))
                    os.makedirs("./Results/" + set_name, exist_ok=True)
                    os.makedirs("./Stats/" + set_name, exist_ok=True)

                    print("#################### CREATING Inner DATASET #######################")
                    loader_train = MyOwnDataset(root='./dataset', mode='train', cases=train_cases, device=device)
                    loader_val = MyOwnDataset(root='./dataset', mode='val', cases=val_cases, device=device)

                    #initialize the created dataset
                    loader_train = DataLoader(loader_train) #opt args: shuffle, batchsize
                    loader_val = DataLoader(loader_val)

                    print("#################### DSS NET parameter #######################")

                    #create hyperparameter
                    latent_dimension = latent_dimension
                    print("Latent space dim : ", latent_dimension)
                    k = k
                    print("Number of updates : ", k)
                    gamma = gamma
                    print("Gamma (loss function) : ", gamma)
                    alpha = alpha
                    print("Alpha (reduction correction) :", alpha)
                    lr = lr
                    print("LR (Learning rate):", lr)


                    print("#################### CREATING NETWORKS #######################")
                    DSS = MyOwnDSSNet(latent_dimension = latent_dimension, k = k, gamma = gamma, alpha = alpha, device=device)
                    # # # DSS = DataParallel(DSS)
                    DSS = DSS.to(device)
                    # # #DSS = DSS.double()

                    print("#################### TRAINING #######################")
                    train_dss = Train_DSS(net=DSS, learning_rate=lr, n_epochs=n_epoch, device=device, set_name=set_name)

                    optimizer, scheduler, epoch, min_val_loss = train_dss.createOptimizerAndScheduler()

                    if restart:
                        optimizer, scheduler, epoch, min_val_loss = train_dss.restart(optimizer, scheduler, path='Model/best_model.pt')

                    GNN = train_dss.trainDSS(loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch, k, n_output)
                    #
                    # # set_trace()
                    #
                    sys.stdout.flush()

                    del DSS, GNN, loader_val, loader_train, optimizer, scheduler
