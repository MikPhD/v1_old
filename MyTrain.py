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
        F = F.cpu().numpy()
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
            print(f'Epoch {epoch} of {self.n_epochs}:')
            time_counter = time.time()

            #--------------------------------TRAIN---------------------------------------------------------
            total_train_loss = 0

            #set net in train mode
            self.net.train()

            for i, train_data in enumerate(loader_train):

                #training operation
                optimizer.zero_grad()
                F, train_loss, loss_dict = self.net(train_data)

                train_loss.sum().backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.e-3)

                optimizer.step()

                total_train_loss += train_loss.sum().item() #somma le loss scalate con 'gamma' per tutto il loader_train per tutti i gusci

                del F, train_loss, loss_dict
                torch.cuda.empty_cache()

                sys.stdout.flush()

            self.hist["loss_train"].append(total_train_loss / len(loader_train)) #relativo alle epoche
            print("Training loss = {:.5e}".format(total_train_loss / len(loader_train)))

            #------------------------------- VALIDATION -----------------------------------------------------
            total_val_loss = 0

            # set net in validation mode
            self.net.eval()

            # validation operation
            with torch.no_grad():
                for val_data in loader_val:
                    F, val_loss, loss_dict = self.net(val_data)
                    total_val_loss += val_loss.sum().item()

            scheduler.step(total_val_loss)

            self.hist["loss_val"].append(total_val_loss / len(loader_val))
            self.training_time = self.training_time + (time.time() - time_counter)
            print("Validation loss = {:.5e}".format(total_val_loss / len(loader_val)))

            torch.cuda.empty_cache()

            sys.stdout.flush()

            #------------------------------- CHECKPOINT ---------------------------------------------------
            intermediate_time = datetime.timedelta(seconds=self.training_time)
            intermediate_time = intermediate_time - datetime.timedelta(microseconds=intermediate_time.microseconds) #avoid to print microseconds

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

            del F, val_loss, loss_dict


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
