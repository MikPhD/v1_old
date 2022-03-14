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
import optuna
from optuna.trial import TrialState
from optuna.pruners import ThresholdPruner, HyperbandPruner
from optuna import TrialPruned
import math
import logging
from optuna.samplers import TPESampler


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', help='epoch number', type=int, default=1)
parser.add_argument('-r', '--restart', type=eval, default=False, choices=[True, False], help='Restart training option')
parser.add_argument('-tcase', '--traincase', help='train cases', nargs="+", default=['40'])
parser.add_argument('-vcase', '--valcase', help='validation cases', nargs="+", default=['40'])
parser.add_argument('-n_out', '--n_output', help='output each n_out epoch', type=int, default=5)

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

#check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on : ', device)

torch.cuda.empty_cache()

def objective(trial):

    torch.cuda.empty_cache()

    print("#################### CREATING Inner DATASET #######################")
    loader_train = MyOwnDataset(root='./dataset', mode='train', cases=train_cases, device=device)
    loader_val = MyOwnDataset(root='./dataset', mode='val', cases=val_cases, device=device)

    #initialize the created dataset
    loader_train = DataLoader(loader_train) #opt args: shuffle, batchsize
    loader_val = DataLoader(loader_val)


    print("#################### DSS NET parameter #######################")
    #create hyperparameter
    latent_dimension = trial.suggest_int("latent_dimension", 1,50)
    print("Latent space dim : ", latent_dimension)
    k = trial.suggest_int("k", 1, 100)
    print("Number of updates : ", k)
    #gamma = (trial.suggest_discrete_uniform("gamma", 0.001, 1, 0.1))
    gamma = 0.1
    print("Gamma (loss function) : ", gamma)
    #alpha = (trial.suggest_discrete_uniform("alpha", 0.001, 20, 0.1))
    alpha = 1e-2
    print("Alpha (reduction correction) :", alpha)
    # lr = (trial.suggest_discrete_uniform("lr", 0.0001, 10, 0.1)) #lr between 0.001 and 0.009
    lr = 3e-3 #fisso
    print("LR (Learning rate):", lr)

    ##create folder for different results ##
    set_name = str(k) + '-' + str(latent_dimension).replace(".", "") + '-' + str(alpha).replace(".", "") + '-' + str(
        lr).replace(".", "")
    print("PARAMETER SET: k:{}, laten_dim:{}, alpha:{}, lr:{}".format(str(k), str(latent_dimension), str(alpha), str(lr)))
    os.makedirs("./Results/" + set_name, exist_ok=True)
    os.makedirs("./Stats/" + set_name, exist_ok=True)


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

    for epoch in range(epoch, n_epoch):
        GNN, validation_loss = train_dss.trainDSS(loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch, k, n_output)

        trial.report(validation_loss, epoch)

        if math.isnan(validation_loss) or math.isinf(validation_loss):
            break

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

        # # use of cuda.memory
        # with open('./Memory_allocated.txt', 'a') as mem_alloc_file:
        #     mem_alloc_file.write(f'memory allocated:{str(torch.cuda.memory_allocated(device))}\n')
        #     mem_alloc_file.write(f'memory reserved:{str(torch.cuda.memory_reserved(device))}\n')
        #     mem_alloc_file.write(f'max_memory allocated: {str(torch.cuda.max_memory_allocated(device))}\n')
        #     mem_alloc_file.write(f'max_memory reserved: {str(torch.cuda.max_memory_reserved(device))}\n')
        #
        # with open('./Memory_summary.txt', 'a') as mem_summ_file:
        #     mem_summ_file.write(f'memory allocated:{str(torch.cuda.memory_summary(device))}\n')

        return validation_loss

    sys.stdout.flush()

    del DSS, GNN, loader_val, loader_train, optimizer, scheduler

################## to be uncommented only when want to log #######################
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "third_optuna"  # Unique identifier of the study.
# storage_name = "sqlite:///{}.db".format(study_name)
##################################################################################

pruner = HyperbandPruner(min_resource=1, max_resource=n_epoch)
study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner,
                            sampler=TPESampler(n_startup_trials=10))
study.optimize(objective)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

fig_optuna = optuna.visualization.plot_contour(study)
fig_optuna.show()
fig_optuna.write_image("./fig_optuna.jpeg")

fig_importance = optuna.visualization.plot_param_importances(study)
fig_importance.show()
fig_importance.write_image("./fig_importance.jpeg")

fig_intermediate = optuna.visualization.plot_intermediate_values(study)
fig_intermediate.show()
fig_intermediate.write_image("./fig_intermediate.jpeg")
