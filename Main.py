from Mydataset import MyOwnDataset
from MyDSS import MyOwnDSSNet
from MyTrain import Train_DSS
from MyCreateData import CreateData
import argparse
import sys
import torch
from torch_geometric.data import DataLoader
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', help='epoch number', type=int, default=10)
parser.add_argument('-r', '--restart', type=eval, default=False, choices=[True, False], help='Restart training option')
parser.add_argument('-tcase', '--traincase', help='train cases', nargs="+", default=['40'])
parser.add_argument('-vcase', '--valcase', help='validation cases', nargs="+", default=['40'])
parser.add_argument('-lr', '--learningrate', help='learning rate', type=float, default=0.01)
parser.add_argument('-n_out', '--n_output', help='output each n_out epoch', type=int, default=1)

args = parser.parse_args()

n_epoch = args.n_epoch
restart = args.restart
train_cases = args.traincase
val_cases = args.valcase
lr = args.learningrate
n_output = args.n_output

# # train_cases = ['40','50','60','70','80','90','100','120','130','140','150']
# train_cases = ['110']
# val_cases = ['110']
# test_cases = ['140']

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

print("#################### DATA ADAPTING FOR GNN #######################")
createdata = CreateData()
createdata.transform(train_cases, 'train')
createdata.transform(val_cases, 'val')

print("#################### CREATING Inner DATASET #######################")
loader_train = MyOwnDataset(root='./dataset', mode='train', cases=train_cases)
loader_val = MyOwnDataset(root='./dataset', mode='val', cases=val_cases)

#initialize the created dataset
loader_train = DataLoader(loader_train) #opt args: shuffle, batchsize
loader_val = DataLoader(loader_val)


print("#################### DSS NET parameter #######################")
#check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on : ', device)

#create hyperparameter
latent_dimension = 10
print("Latent space dim : ", latent_dimension)
k = 30
print("Number of updates : ", k)
gamma = 0.9
print("Gamma (loss function) : ", gamma)
alpha = 1.e-3
print("Alpha (reduction correction) :", alpha)
print("LR (Learning rate):", lr)


print("#################### CREATING NETWORKS #######################")
DSS = MyOwnDSSNet(latent_dimension = latent_dimension, k = k, gamma = gamma, alpha = alpha, device=device)
# # # DSS = DataParallel(DSS)
DSS = DSS.to(device)
# # #DSS = DSS.double()

print("#################### TRAINING #######################")
train_dss = Train_DSS(net=DSS, learning_rate=lr, n_epochs=n_epoch, device=device)

optimizer, scheduler, epoch, min_val_loss = train_dss.createOptimizerAndScheduler()

if restart:
    optimizer, scheduler, epoch, min_val_loss = train_dss.restart(optimizer, scheduler, path='Model/best_model.pt')

GNN = train_dss.trainDSS(loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch, k, n_output)
#
# # set_trace()
#
sys.stdout.flush()
