from Mydataset import MyOwnDataset
from MyDSS import MyOwnDSSNet
from MyTrain import Train_DSS
from MyCreateData import CreateData
import argparse
import sys
import torch
from torch_geometric.data import DataListLoader
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', help='epoch number', type=int, default=10)
parser.add_argument('-r', '--restart', type=eval, default=False, choices=[True, False], help='Restart training option')
args = parser.parse_args()

n_epoch = args.n_epoch
restart = args.restart

# train_cases = ['40','50','60','70','80','90','100','120','130','140','150']
train_cases = ['110']
val_cases = ['110']
test_cases = ['140']

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
k = 2
print("Number of updates : ", k)
gamma = 0.9
print("Gamma (loss function) : ", gamma)
alpha = 1.e-3
print("Alpha (reduction correction) :", alpha)


print("#################### CREATING NETWORKS #######################")
DSS = MyOwnDSSNet(latent_dimension = latent_dimension, k = k, gamma = gamma, alpha = alpha, device=device)
# # # DSS = DataParallel(DSS)
DSS = DSS.to(device)
# # #DSS = DSS.double()
#
print("#################### TRAINING #######################")
train_dss = Train_DSS(net=DSS, learning_rate=0.01, n_epochs=n_epoch, device=device)

optimizer, scheduler, epoch, min_val_loss = train_dss.createOptimizerAndScheduler()

if restart:
    optimizer, scheduler, epoch, min_val_loss = train_dss.restart(optimizer, scheduler, path='Model/best_model.pt')

GNN = train_dss.trainDSS(loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch, k)
#
# # set_trace()
#
sys.stdout.flush()
