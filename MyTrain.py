import numpy as np
import torch
import time
import sys
import os
from MyPlot import Plot
import datetime


class Train_DSS:
    def __init__(self, net, learning_rate = 0.01, n_epochs = 20, device = "cpu", set_name=""):

        #Initialize training parameters
        self.set_name = set_name
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.net = net
        self.device = device
        self.training_time = 0
        self.hist = {"loss_train":[], "loss_val":[]}

    def createOptimizerAndScheduler(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.lr, weight_decay=0)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=50,
                                                               min_lr=0.0001, verbose=True)
        min_val_loss = 1.e-1
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

    def restart(self, optimizer, scheduler, path):
        optimizer, scheduler, epoch, min_val_loss = self.load_model(path, optimizer, scheduler)

        return optimizer, scheduler, epoch, min_val_loss

    def export_results(self, F, k, epoch, last=False):
        F = F[str(k)].cpu().numpy()
        if last:
            np.save("./Results/" + self.set_name + "/results.npy", F)
        else:
            np.save("./Results/" + self.set_name + "/results" + str(epoch + 1) + ".npy", F)

        ### Save log files ###
        with open('./Stats/' + self.set_name + '/loss_train_log.txt', 'w') as f_loss_train:
            f_loss_train.write(str(self.hist["loss_train"]))

        with open('./Stats/' + self.set_name + '/loss_val_log.txt', 'w') as f_loss_val:
            f_loss_val.write(str(self.hist["loss_val"]))

        ### Close log files ###
        f_loss_train.close()
        f_loss_val.close()

        ## Save plot ##
        try:
            MyPlot = Plot(self.set_name)
            MyPlot.plot_loss()

            if last:
                MyPlot.plot_results("")
                print("Last Plot Saved!")
            else:
                MyPlot.plot_results(epoch + 1)
                print("Intermediate Plot Saved!")
        except:
            print("Problema di PLOT!")


    def trainDSS(self, loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch_in, k, n_output):
        for epoch in range(epoch_in, self.n_epochs):
            print(f'Epoch: {epoch} of {self.set_name}:')
            time_counter = time.time()

            #--------------------------------TRAIN---------------------------------------------------------
            total_train_loss = 0

            #set net in train mode
            self.net.train()

            for i, train_data in enumerate(loader_train):

                #training operation
                optimizer.zero_grad()
                F, train_loss_first, train_loss_second, loss_dict1, loss_dict2 = self.net(train_data, epoch, self.n_epochs)

                train_loss_first.sum().backward(retain_graph=True)
                grad1 = []
                for p in self.net.parameters():
                    grad1.append(p.grad.data.clone())
                grad1 = torch.cat([torch.flatten(grad.cpu()) for grad in grad1])
                optimizer.zero_grad()

                train_loss_second.sum().backward(retain_graph=True)
                grad2 = []
                for p in self.net.parameters():
                    grad2.append(p.grad.data.clone())
                grad2 = torch.cat([torch.flatten(grad.cpu()) for grad in grad2])
                optimizer.zero_grad()
                #
                v1v1 = torch.dot(torch.t(grad1), grad1)
                v1v2 = torch.dot(torch.t(grad1), grad2)
                v2v2 = torch.dot(torch.t(grad2), grad2)
                #
                diff = torch.sub(grad2, grad1)
                #
                if v1v2 >= v2v2:
                    alpha = 0
                elif v1v2 >= v1v1:
                    alpha = 1
                else:
                    alpha = torch.div(torch.dot(grad2, torch.t(diff)), torch.pow(torch.norm(diff), 2))

                print(f'alpha: {alpha}')

                loss_tot = alpha *(train_loss_first) + (1-alpha) * (train_loss_second)

                loss_tot.sum().backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.e-3)  # da riattivare
                optimizer.step()

                total_train_loss += (alpha) * train_loss_first.sum().item() + (1-alpha) * train_loss_second.sum().item()

                del F, train_loss_first, train_loss_second, loss_dict1, loss_dict2
                torch.cuda.empty_cache()

                sys.stdout.flush()

            self.hist["loss_train"].append(float(total_train_loss) / len(loader_train))
            print("Training loss = {:.5e}".format(total_train_loss / len(loader_train)))

            #------------------------------- VALIDATION -----------------------------------------------------
            total_val_loss = 0

            # set net in validation mode
            self.net.eval()

            # validation operation
            with torch.no_grad():
                for val_data in loader_val:
                    F, val_loss_first, val_loss_second, loss_dict1, loss_dict2 = self.net(val_data, epoch, self.n_epochs)
                    total_val_loss += (alpha) * val_loss_first.sum().item() + (1-alpha) * val_loss_second.sum().item()

            scheduler.step(total_val_loss)

            self.hist["loss_val"].append(float(total_val_loss) / len(loader_val))
            self.training_time = self.training_time + (time.time() - time_counter)
            print("Validation loss = {:.5e}".format(total_val_loss / len(loader_val)))

            torch.cuda.empty_cache()

            sys.stdout.flush()

            #------------------------------- CHECKPOINT ---------------------------------------------------
            intermediate_time = datetime.timedelta(seconds=self.training_time)

            if total_val_loss / len(loader_val) <= min_val_loss:

                checkpoint = {
                    'epoch': epoch + 1,
                    'min_val_loss': total_val_loss / len(loader_val),
                    'state_dict': self.net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss_train': self.hist["loss_train"],
                    'loss_val': self.hist["loss_val"],
                    'training_time': self.training_time
                }

                # save model
                self.save_model(checkpoint, dirName="Model", model_name="best_model")
                min_val_loss = total_val_loss / len(loader_val)

                print(f"Training finished, took {intermediate_time}, MODEL SAVED")

            else:
                print(f"Training finished, took {intermediate_time}s")


            #------------------------------------EXPORT INTERMEDIATE RESULTS -----------------------------------------
            if int(epoch + 1) % n_output == 0:
                self.export_results(F, k, epoch)

            del F, val_loss_first, val_loss_second, loss_dict1, loss_dict2


        #--------------------------------------- FINAL OPERATION -------------------------------------------------
        ## Save last model ##
        checkpoint = {
            'epoch': epoch + 1,
            'min_val_loss': total_val_loss / len(loader_val),
            'state_dict': self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_train': self.hist["loss_train"],
            'loss_val': self.hist["loss_val"],
            'training_time': self.training_time
        }
        self.save_model(checkpoint, dirName="Model", model_name="best_model_final")

        return self.net
