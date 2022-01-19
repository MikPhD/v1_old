import time
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

class Stats:
    def __init__(self, k, len_train, len_val, hist, training_time):
        self.hist = hist
        self.len_train = len_train
        self.len_val = len_val
        self.k = k
        self.total_train_loss = 0
        self.final_train_loss = 0
        self.total_val_loss = 0
        self.final_val_loss = 0

        self.time_counter = time.time()
        self.training_time = training_time

    def compute_loss_train(self, train_loss, loss_dict):
        self.total_train_loss += train_loss.sum().item()
        self.final_train_loss += loss_dict[str(self.k)].sum().item()

        return self.final_train_loss

    def compute_loss_val(self, val_loss, loss_dict):
        self.total_val_loss += val_loss.sum().item()
        self.final_val_loss += loss_dict[str(self.k)].sum().item()

        return self.final_val_loss

    def hist_log(self):
        self.hist["loss_train"].append(self.final_train_loss / self.len_train)
        self.hist["loss_val"].append(self.final_val_loss / self.len_val)

        return self.hist


    def print_stats(self, epoch, upd = False):
        print("Epoch: ", epoch + 1)
        print("Training loss = {:.5e}".format(self.total_train_loss / self.len_train))
        print("Validation loss = {:.5e}".format(self.total_val_loss / self.len_val))

        self.training_time = self.training_time + (time.time() - self.time_counter)
        if upd:
            print("Training finished, took {:.2f}s, MODEL UPDATED".format(self.training_time))
        else:
            print("Training finished, took {:.2f}s,".format(self.training_time))

        return self.training_time

    def save_output(self, epoch, F):
        ### SAVE RESULTS ###
        F_fin = F[str(self.k)].cpu().numpy()
        np.save("./Results/results" + str(epoch) + ".npy", F_fin)

        ### Delete previous log files ###
        if os.path.exists("Stats/loss_val_log.txt"):
            os.remove('Stats/loss_val_log.txt')
            os.remove('Stats/loss_train_log.txt')

        ### Save new log files ###
        with open('Stats/loss_train_log.txt', 'w') as f_loss_train:
            f_loss_train.write(str(self.hist["loss_train"]))

        with open('Stats/loss_val_log.txt', 'w') as f_loss_val:
            f_loss_val.write(str(self.hist["loss_val"]))

        ### Close log files ###
        f_loss_train.close()
        f_loss_val.close()

    def plot_loss(self):
        ### Open log files
        with open('Stats/loss_train_log.txt', 'r') as f_train:
            mydata_train = ast.literal_eval(f_train.read())
        with open('Stats/loss_val_log.txt', 'r') as f_val:
            mydata_val = ast.literal_eval(f_val.read())

        ### define axis and data ###
        dt = 1
        x = np.arange(0, len(mydata_train), dt)
        y_train = mydata_train
        y_val = mydata_val

        plt.plot(x, y_train, x, y_val)
        plt.savefig("Stats/plot_loss.jpg")

        ### Close Files ###
        f_train.close()
        f_val.close()
