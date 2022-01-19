from fenics import *
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from pdb import set_trace

class CreateData:
    def __init__(self):
        self.comm = MPI.comm_world
        set_log_level(40)
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["std_out_all_processes"] = False

    def transform(self, cases, mode):
        for h in cases:
            #####initializing###
            print("Elaborazione case: ", h, " in modalità: ", mode)

            ################# inizio lettura file ##########################
            ######### lettura mesh #########
            mesh = Mesh()
            mesh_file = "../Dataset/" + str(h) + "/Mesh.h5"
            with HDF5File(self.comm, mesh_file, "r") as h5file:
                h5file.read(mesh, "mesh", False)
                facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                h5file.read(facet, "facet")

            ###### lettura risultati ########
            VelocityElement2 = VectorElement("CG", mesh.ufl_cell(), 2)
            PressureElement2 = FiniteElement("CG", mesh.ufl_cell(), 1)
            Space2 = FunctionSpace(mesh, VelocityElement2 * PressureElement2)

            u_glob2 = Function(Space2)
            f_glob2 = Function(Space2)

            with HDF5File(self.comm, "../Dataset/" + str(h) + "/Results.h5", "r") as h5file:
                h5file.read(u_glob2, "mean")
                h5file.read(f_glob2, "forcing")

            # è necessario perchè lavoro sugli indici dei vertici della mesh!

            ############# CG 2 to CG 1 ##############
            VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
            PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
            Space = FunctionSpace(mesh, VelocityElement * PressureElement)

            u_glob = project(u_glob2, Space)
            f_glob = project(f_glob2, Space)

            ################### End lettura file ####################################

            ################### Creazione elementi dataset ##########################
            ######### connectivity mesh ###########
            mesh.init()
            mesh_topology = mesh.topology()
            mesh_connectivity = mesh_topology(1, 0)
            C = [] ##connection list
            D = [] ##distances between connection
            for i in range(mesh.num_edges()):
                connection = np.array(mesh_connectivity(i)).astype(int)
                coord_vert1 = mesh.coordinates()[connection[0]]
                coord_vert2 = mesh.coordinates()[connection[1]]
                distancex = coord_vert1[0]-coord_vert2[0]
                distancey = coord_vert1[1]-coord_vert2[1]

                C.append(list(connection))
                D.append([distancex, distancey])

            ############## mean flow e forcing #####################
            ########### mappatura attraverso coordinate ############
            ###### vertices itera sugli indici da 0 in poi #########

            # coord = []
            U_P = []  ##u_mean + pressure(terza componente)
            F = []  ##forcing
            for v in vertices(mesh):
                # coord.append(np.array(mesh.coordinates()[v.index()]).tolist())
                U_P.append(np.array(u_glob(mesh.coordinates()[v.index()])).tolist())
                F.append(np.array(f_glob(mesh.coordinates()[v.index()])).tolist())

            ###### remove useless component -> z component#####
            for j in F:
                del j[2]
            ###### remove pressure ########
            # for i in U_P:
            #     del i[2]

        ######################### Fine creazione elementi############################

        ###################### Salvataggio file Numpy ##############################
            os.makedirs("./dataset/raw/" + mode + "/" + h, exist_ok=True)
            specific_dir = "./dataset/raw/" + mode + "/" + h
            np.save(specific_dir + "/C.npy", C)
            np.save(specific_dir + "/D.npy", D)
            np.save(specific_dir + "/U_P.npy", U_P)
            np.save(specific_dir + "/F.npy", F)
            # np.save(specific_dir + "/coord.npy", coord)
            np.save(specific_dir + "/re.npy", int(h))
        ################# Fine salvataggio file ##################################

        ################# Print interface ########################################
        print("Trasformazione file di " + mode + " completata!")

