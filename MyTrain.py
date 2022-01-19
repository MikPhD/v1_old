import numpy as np
import torch
import time
import sys
import torchvision
import torch.nn as nn
import os
from MyStats import Stats
from progress.bar import Bar


class Train_DSS:
    def __init__(self, net, learning_rate = 0.01, n_epochs = 20, device = "cpu"):

        #Initialize training parameters
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.net = net
        self.device = device
        self.training_time = 0
        self.hist = {"loss_train":[], "loss_val":[]}

    def createOptimizerAndScheduler(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.1)
        min_val_loss = 0
        epoch = 0
        return optimizer, scheduler, epoch, min_val_loss

    def save_model(self, state, dirName="Model", model_name="best_model"):

        if not os.path.exists(dirName):
            os.makedirs(dirName)
        model_name = "{}.pt".format(model_name)
        save_path = os.path.join(dirName,model_name)
        path = open(save_path, mode="wb")
        torch.save(state, path)
        path.close()

    def load_model(self, path, optimizer, scheduler):

        #Load checkpoint
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        min_val_loss = checkpoint['min_val_loss']
        self.hist['loss_train'] = checkpoint['loss_train']
        self.hist['loss_val'] = checkpoint['loss_val']
        self.training_time = checkpoint['training_time']

        return optimizer, scheduler, checkpoint['epoch'], min_val_loss

    def restart(self, path='./'):
        optimizer, scheduler, epoch, min_val_loss = self.load_model(path, optimizer, scheduler)

        return optimizer, scheduler, epoch, min_val_loss

    def trainDSS(self, loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch_in, k):
        with Bar("Training...", min=epoch_in, max=self.n_epochs) as bar:
            for epoch in range(epoch_in, self.n_epochs):
                bar.next()
                # Initialize Stats class
                stats = Stats(k, len(loader_train), len(loader_val), self.hist, self.training_time)
                final_train_loss, final_val_loss = 0, 0

                ##################### Training STEP #######################
                self.net.train()

                for i, train_data in enumerate(loader_train):
                    optimizer.zero_grad()
                    train_data = train_data.to(self.device)
                    F, train_loss, loss_dict = self.net(train_data)

                    train_loss.sum().backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.e-2) #da riattivare
                    optimizer.step()

                    stats.compute_loss_train(train_loss, loss_dict)

                    del F, train_loss, loss_dict
                    torch.cuda.empty_cache()
                    sys.stdout.flush()


                ################ Validation STEP #####################
                self.net.eval()

                with torch.no_grad():
                    for val_data in loader_val:
                        val_data = val_data.to(self.device)
                        F, val_loss, loss_dict = self.net(val_data)

                        final_val_loss = stats.compute_loss_val(val_loss, loss_dict)
                scheduler.step()
                sys.stdout.flush()


                ############### Stats## ##################

                self.hist = stats.hist_log()

                ############### Checkpoint ##########################

                if final_val_loss / len(loader_val) <= min_val_loss:
                    self.training_time = stats.print_stats(epoch, upd=True)

                    min_val_loss = final_val_loss / len(loader_val)

                    checkpoint = {
                        'epoch': epoch + 1,
                        'min_val_loss': final_val_loss / len(loader_val),
                        'state_dict': self.net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss_train': self.hist["loss_train"],
                        'loss_val': self.hist["loss_val"],
                        'training_time': self.training_time
                    }
                    # save model
                    self.save_model(checkpoint, dirName="Model", model_name="best_model")

                else:
                    self.training_time = stats.print_stats(epoch)

                ############## Save Output ############################
                if int(epoch) % 10000 == 0:
                    stats.save_output(epoch, F)

        ### Final step before close ###
        checkpoint = {
            'epoch': epoch + 1,
            'min_val_loss': final_val_loss / len(loader_val),
            'state_dict': self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_train': self.hist["loss_train"],
            'loss_val': self.hist["loss_val"],
            'training_time': self.training_time
        }
        # save model
        self.save_model(checkpoint, dirName="Model", model_name="best_model_normal_final")

        stats.save_output(epoch, F)

        stats.plot_loss()

        return self.net

# def loss_function(U, edge_index, edge_attr, y): ##non utilizzata --> utilizzo mse_loss
#
#     B0 = y[:,0].reshape(-1,1)
#     B1 = y[:,1].reshape(-1,1)
#     # B2 = y[:,2].reshape(-1,1)
#
#     p1 = (1 - B1)*(-B0) + B1*(U)
#
#     from_ = edge_index[0,:].reshape(-1,1).type(torch.int64)
#     to_ = edge_index[1,:].reshape(-1,1).type(torch.int64)
#
#     u_i = torch.gather(U, 0, from_)
#     u_j = torch.gather(U, 0, to_)
#
#     F_bar = edge_attr*(u_i-u_j)
#     M = U*0
#     F_bar_sum = M.scatter_add(0,from_,F_bar)
#
#     residuals = p1 + F_bar_sum
#
#     return torch.mean(residuals**2)
#
# def corrcoef(x,y):
#     vx = x - torch.mean(x)
#     vy = y - torch.mean(y)
#     cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
#     return cost
