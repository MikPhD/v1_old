from dolfin import *
import numpy as np
import os
import os.path as osp
from io import IOBase
import sys
import math
import torch
import torch_geometric
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data, DataLoader
# from torch_geometric.nn import DataParallel
from pdb import set_trace

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, mode, cases, transform=None, pre_transform=None):
        self.mode = mode
        self.cases = cases
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

        ##qui viene eseguito tutto il codice (download, process ecc...)

        if self.mode == 'train':
            path = self.processed_paths[0]
        elif self.mode == 'val':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        files = []
        for h in self.cases:
            dir = self.mode + "/" + h + "/"

            files.append(dir + "C.npy")
            # files.append(self.mode + h + "coord.npy")
            files.append(dir + "D.npy")
            files.append(dir + "F.npy")
            files.append(dir + "re.npy")
            files.append(dir + "U_P.npy")
        # set_trace()
        return files

    @property
    def processed_dir(self):
        return self.root + "/processed/"

    @property
    def processed_file_names(self):
        return ['data_train.pt', 'data_val.pt', 'data_test.pt']

    def download(self):
        print('File di input non presenti o non completi!!')


    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        directory = self.root + "/raw/" + self.mode + "/"

        for j in self.cases:
            print("Taking Data at Re n.: ", j, "in ", self.mode, " mode.")
            input_dir = directory + str(j)

            C = np.load(input_dir + '/C.npy')  #connection list
            D = np.load(input_dir + '/D.npy')  #edge attr - distances
            U_P = np.load(input_dir + '/U_P.npy')  #campo medio + pressione
            F = np.load(input_dir + '/F.npy')  #forzaggio
            Re = np.load(input_dir + '/re.npy')  #Reynolds number

            U_P_Re = np.insert(U_P, 3, Re, axis=1)

            edge_index = torch.tensor(C, dtype=torch.long)
            edge_attr = torch.tensor(D, dtype=torch.float)

            x = torch.tensor(U_P_Re, dtype=torch.float)
            y = torch.tensor(F, dtype=torch.float)

            data = Data(x = x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)
            data_list.append(data)


            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        if self.mode == 'train':
            torch.save(self.collate(data_list), self.processed_paths[0])
        elif self.mode == 'val':
            torch.save(self.collate(data_list), self.processed_paths[1])
        else:
            torch.save(self.collate(data_list), self.processed_paths[2])
