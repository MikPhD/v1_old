from Mydataset import MyOwnDataset
from MyDSS import MyOwnDSSNet
from MyTrain import Train_DSS
from MyCreateData import CreateData
import argparse
import sys
import torch
import os
import shutil
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch_tot', help='epoch number', type=int, default=3)
parser.add_argument('-r', '--restart', type=eval, default=True, choices=[True, False], help='Restart training option')
parser.add_argument('-tcase', '--traincase', help='train cases', nargs="+", default=['40'])
parser.add_argument('-vcase', '--valcase', help='validation cases', nargs="+", default=['40'])
parser.add_argument('-n_out', '--n_output', help='output each n_out epoch', type=int, default=1)

args = parser.parse_args()

n_epoch_tot = args.n_epoch_tot
restart = args.restart
train_cases = args.traincase
val_cases = args.valcase
n_output = args.n_output

# train_cases = ['40','50','60','70','80','90','100','120','130','140','150']
train_cases = ['40', '50']
val_cases = ['40']
# test_cases = ['110']

# ## Copy Mesh file in Results - needed for plot ##
# for i in val_cases:
#     src = os.path.join("../Dataset", i, "Mesh.h5")
#     dst = "./Results/Mesh.h5"
#     shutil.copyfile(src, dst)

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on : ', device)

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
createdata = CreateData()
createdata.transform(train_cases, 'train')
createdata.transform(val_cases, 'val')

print("#################### CREATING Inner DATASET #######################")
loader_train = MyOwnDataset(root='./dataset', mode='train', cases=train_cases, device=device)
loader_val = MyOwnDataset(root='./dataset', mode='val', cases=val_cases, device=device)

#initialize the created dataset
loader_train = DataLoader(loader_train, shuffle=True) #opt args: shuffle, batchsize
loader_val = DataLoader(loader_val)

print("#################### DSS NET parameter #######################")
k = 1
latent_dimension = 18
gamma = 0.1
alpha = 1e-2
lr = 3e-3

#create hyperparameter
print("Latent space dim : ", latent_dimension)
print("Number of updates : ", k)
print("Gamma (loss function) : ", gamma)
print("Alpha (reduction correction) :", alpha)
print("LR (Learning rate):", lr)

# ##create folder for different results ##
# set_name = str(k) + '-' + str(latent_dimension).replace(".", "") + '-' + str(alpha).replace(".", "") + '-' + str(
#     lr).replace(".", "")
# print("PARAMETER SET: k:{}, laten_dim:{}, alpha:{}, lr:{}".format(str(k), str(latent_dimension), str(alpha), str(lr)))
# os.makedirs("./Results/" + set_name, exist_ok=True)
# os.makedirs("./Stats/" + set_name, exist_ok=True)


print("#################### CREATING NETWORKS #######################")
DSS = MyOwnDSSNet(latent_dimension=latent_dimension, k=k, gamma=gamma, alpha=alpha, device=device)
# # # DSS = DataParallel(DSS)
DSS = DSS.to(device)
# # #DSS = DSS.double()

print("#################### TRAINING #######################")
train_dss = Train_DSS(net=DSS, learning_rate=lr, n_epochs_tot=n_epoch_tot, device=device)

optimizer, scheduler, epoch, min_val_loss = train_dss.createOptimizerAndScheduler()

if restart:
    optimizer, scheduler, epoch, min_val_loss = train_dss.restart(optimizer, scheduler,
                                                                  path='Model/best_model.pt')

GNN = train_dss.trainDSS(loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch, k,
                         n_output)
#
sys.stdout.flush()

del DSS, GNN, loader_val, loader_train, optimizer, scheduler