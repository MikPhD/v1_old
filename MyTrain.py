import numpy as np
import torch
import time
import sys
import os
from MyPlot import Plot
import ast
import matplotlib.pyplot as plt
import torch.nn as nn


from progress.bar import Bar


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
                                                               min_lr=0.001, verbose=True)
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


    def trainDSS(self, loader_train, loader_val, optimizer, scheduler, min_val_loss, epoch_in, k, n_output):
        for epoch in range(epoch_in, self.n_epochs):
            time_counter = time.time()

            total_train_loss, running_loss = 0, 0
            final_loss, running_final_loss = 0, 0
            # rmse, running_rmse = 0, 0

            #set net in train mode
            self.net.train()

            for i, train_data in enumerate(loader_train):

                #training operation
                optimizer.zero_grad()
                F, train_loss_first, train_loss_second, loss_dict1, loss_dict2 = self.net(train_data, epoch, self.n_epochs)
                # sol_lu = train_data.x.to(U[str(k)].device)
                # sol_lu = torch.cat([data.x for data in train_data]).to(U[str(k)].device)
                # sol_lu = torch.cat([(next(iter(train_data))).x]).to(U[str(k)].device)

                train_loss_first.sum().backward(retain_graph=True)
                l1_parameter = {}
                for name, params in self.net.named_parameters():
                    l1_parameter[name] = params.grad.clone()

                optimizer.zero_grad()
                train_loss_second.sum().backward()
                l2_parameter = {}
                for name, params in self.net.named_parameters():
                    l2_parameter[name] = params.grad.clone()

                l3_parameter = {}
                alpha = 0.5
                def mod_param(key, tensor1, tensor2):
                    print(f'key:{key}, index: {index}')
                    norm_loss1 = torch.norm(tensor1)
                    sq_norm_loss1 = torch.pow(norm_loss1, 2)

                    norm_loss2 = torch.norm(tensor2)
                    sq_norm_loss2 = torch.pow(norm_loss2, 2)

                    loss1_loss2 = torch.dot(tensor1, tensor2)

                    norm_sq_loss1loss2 = torch.pow(torch.norm(torch.sub(tensor1, tensor2)),2)

                    # if loss1_loss2 < min(sq_norm_loss1, sq_norm_loss2):
                    #     alpha = torch.div(torch.dot(tensor1, torch.sub(tensor1, tensor2)), norm_sq_loss1loss2)
                    #
                    # elif min(norm_loss1, norm_loss2) == norm_loss1:
                    #     alpha = 1
                    # elif min(norm_loss1, norm_loss2) == norm_loss2:
                    #     alpha = 0

                    # l3_parameter[key] = (alpha) * l1_parameter[key] + (1 - alpha) * tensor2

                    first_term = torch.mul(torch.div(torch.sub(sq_norm_loss1, loss1_loss2), torch.sub(sq_norm_loss2 + sq_norm_loss1, 2*loss1_loss2)), tensor2)
                    second_term = torch.mul(torch.div(torch.sub(sq_norm_loss2, loss1_loss2), torch.sub(sq_norm_loss2 + sq_norm_loss1, 2*loss1_loss2)), tensor1)

                    l3_parameter[key] = first_term + second_term
                    # return alpha


                for key in l1_parameter:
                    if l1_parameter[key].dim() > 1:
                        for index, tensor in enumerate(l1_parameter[key]):
                            mod_param(key, l1_parameter[key][index], l2_parameter[key][index])

                    else:
                        mod_param(key, l1_parameter[key], l2_parameter[key])

                optimizer.zero_grad()

                #reassign paramter
                for name, params in self.net.named_parameters():
                    params.grad.data.copy_(l3_parameter[name])

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.e-3)  # da riattivare
                optimizer.step()

                total_train_loss += (1-alpha) * train_loss_first.sum().item() + (alpha) * train_loss_second.sum().item()
                final_loss += (1-alpha) * loss_dict1[str(k)].sum().item() + (alpha) * loss_dict2[str(k)].sum().item()

                running_loss += (1-alpha) * train_loss_first.sum().item() + (alpha) * train_loss_second.sum().item()
                running_final_loss += (1-alpha) * loss_dict1[str(k)].sum().item() + (alpha) * loss_dict2[str(k)].sum().item()

                ##print during training set cycle loop
                if (i + 1) % (len(loader_train) // 5 + 1) == 0:
                    print(
                        "Epoch {}, {:d}% \t train_loss: {:.5e}".format(
                            epoch + 1,
                            int(100 * (i + 1) / len(loader_train)),
                            running_loss / (len(loader_train) // 1)))

                    running_loss = 0.0
                    running_final_loss = 0

                del F, train_loss_first, train_loss_second, loss_dict1, loss_dict2
                torch.cuda.empty_cache()

                sys.stdout.flush()

            self.hist["loss_train"].append(final_loss / len(loader_train))
            print("Training loss = {:.5e}".format(total_train_loss / len(loader_train)))

            total_val_loss = 0
            final_loss_val = 0

            # set net in validation mode
            self.net.eval()

            # validation operation
            with torch.no_grad():
                for val_data in loader_val:
                    F, val_loss_first, val_loss_second, loss_dict1, loss_dict2 = self.net(val_data, epoch, self.n_epochs)
                    # sol_lu = val_data.x.to(U[str(k)].device) #da riattivare
                    total_val_loss += (1-alpha) * val_loss_first.sum().item() + (alpha) * val_loss_second.sum().item()
                    final_loss_val += (1-alpha) * loss_dict1[str(k)].sum().item() + (alpha) * loss_dict2[str(k)].sum().item()
                    # corr_val += corrcoef(U[str(k)], sol_lu)
                    # rmse_val += torch.sqrt(torch.mean(U[str(k)] - sol_lu) ** 2)

            scheduler.step(total_val_loss)

            self.hist["loss_val"].append(final_loss_val / len(loader_val))
            self.training_time = self.training_time + (time.time() - time_counter)
            print("Validation loss = {:.5e}".format(total_val_loss / len(loader_val)))


            torch.cuda.empty_cache()

            sys.stdout.flush()

            if final_loss_val / len(loader_val) <= min_val_loss:

                checkpoint = {
                    'epoch': epoch + 1,
                    'min_val_loss': final_loss_val / len(loader_val),
                    'state_dict': self.net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss_train': self.hist["loss_train"],
                    'loss_val': self.hist["loss_val"],
                    'training_time': self.training_time
                }
                # save model
                self.save_model(checkpoint, dirName="Model", model_name="best_model")
                min_val_loss = final_loss_val / len(loader_val)
                print("Training finished, took {:.2f}s, MODEL SAVED".format(self.training_time))

            else:
                print("Training finished, took {:.2f}s".format(self.training_time))

            if (int(epoch + 1) % n_output == 0) and (int(epoch + 1 != self.n_epochs)):
                F_fin = F[str(k)].cpu().numpy()
                np.save("./Results/" + self.set_name + "/results" + str(epoch + 1) + ".npy", F_fin)

                ### Save new log files ###
                with open('Stats/' + self.set_name + '/loss_train_log.txt', 'w') as f_loss_train:
                    f_loss_train.write(str(self.hist["loss_train"]))

                with open('Stats/' + self.set_name + '/loss_val_log.txt', 'w') as f_loss_val:
                    f_loss_val.write(str(self.hist["loss_val"]))

                ### Close log files ###
                f_loss_train.close()
                f_loss_val.close()

                ## Save plot training ##
                MyPlot = Plot(self.set_name)
                MyPlot.plot_loss()
                MyPlot.plot_results(epoch + 1)
                # try:
                #     MyPlot = Plot(self.set_name)
                #     MyPlot.plot_loss()
                #     MyPlot.plot_results(epoch + 1)
                # except:
                #     print("errore di plot")

                print("Intermediate Plot Saved!")
                del F, val_loss_first, val_loss_second, loss_dict1, loss_dict2

            if int(epoch + 1) == self.n_epochs:
                F_fin = F[str(k)].cpu().numpy()
                np.save("./Results/" + self.set_name + "/results.npy", F_fin)

                ### Save new log files ###
                with open('Stats/' + self.set_name + '/loss_train_log.txt', 'w') as f_loss_train:
                    f_loss_train.write(str(self.hist["loss_train"]))

                with open('Stats/' + self.set_name + '/loss_val_log.txt', 'w') as f_loss_val:
                    f_loss_val.write(str(self.hist["loss_val"]))

                ### Close log files ###
                f_loss_train.close()
                f_loss_val.close()

                ## Save plot training ##
                try:
                    MyPlot = Plot(self.set_name)
                    MyPlot.plot_loss()
                    MyPlot.plot_results("")
                except:
                    print("errore di plot")


                print("Final Results Saved")
                del F, val_loss_first, val_loss_second, loss_dict1, loss_dict2


        ## Save last model ##
        checkpoint = {
            'epoch': epoch + 1,
            'min_val_loss': final_loss_val / len(loader_val),
            'state_dict': self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_train': self.hist["loss_train"],
            'loss_val': self.hist["loss_val"],
            'training_time': self.training_time
        }
        self.save_model(checkpoint, dirName="Model", model_name="best_model_normal_final")

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
