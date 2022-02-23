import matplotlib.pyplot as plt
from fenics import *
import matplotlib.tri as tri
import argparse
from pdb import set_trace
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from collections import OrderedDict
import ast
import numpy as np
from matplotlib.ticker import ScalarFormatter


class Plot:
    def __init__(self):
        self.cmaps = OrderedDict()
        self.cmaps['Perceptually Uniform Sequential'] = ['plasma_r']

    def plot_loss(self):
        ### Open log files
        with open('Stats' + '/loss_train_log.txt', 'r') as f_train:
            mydata_train = ast.literal_eval(f_train.read())
        with open('Stats' + '/loss_train_log.txt', 'r') as f_val:
            mydata_val = ast.literal_eval(f_val.read())

        ### define axis and data ###
        dt = 1
        x = np.arange(0, len(mydata_train), dt)
        y_train = mydata_train
        y_val = mydata_val

        plt.plot(x, y_train, x, y_val)
        plt.savefig("Stats/"  + "plot_loss.jpg")

        ### Close Files ###
        f_train.close()
        f_val.close()

    def mesh2triang(self, mesh):
        xy = mesh.coordinates()
        return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    def plot(self, obj):
        plt.gca().set_aspect('equal')
        if isinstance(obj, Function):
            mesh = obj.function_space().mesh()
            if obj.vector().size() == mesh.num_cells():
                C = obj.vector().array()
                plt.tripcolor(self.mesh2triang(mesh), C)
            else:
                C = obj.compute_vertex_values(mesh)
                plt.tripcolor(self.mesh2triang(mesh), C, shading='gouraud')
        elif isinstance(obj, Mesh):
            plt.triplot(self.mesh2triang(obj), color='k')

    def plot_results(self, n_epoch = ""):

        ####### loading mesh ########
        mesh = Mesh()
        mesh_file = "./Results/Mesh.h5"
        with HDF5File(MPI.comm_world, mesh_file, "r") as h5file:
            h5file.read(mesh, "mesh", False)
            facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
            h5file.read(facet, "facet")

        ####### initializing holder ########
        VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
        PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
        Space = FunctionSpace(mesh, VelocityElement * PressureElement)

        F = Function(Space)

        u, f = F.split(deepcopy=True)
        # f.set_allow_extrapolation(True)

        # with HDF5File(MPI.comm_world, "../Dataset/" + case_num + "/Results.h5", "r") as h5file:
        #     h5file.read(f, "mean")
        #     h5file.read(f, "forcing")

        # ####### loading forcing from GNN ################
        F_gnn = np.load('./Results/results.npy').flatten()
        # mesh_points = np.load('./Results/mesh_points.npy').tolist()
        mesh_points = mesh.coordinates().tolist()

        dofs_coordinates_prev = Space.sub(0).collapse().tabulate_dof_coordinates().tolist()

        for i, x in enumerate(mesh_points):
            index = dofs_coordinates_prev.index(x)
            u.vector()[(index)] = F_gnn[i*2]
            u.vector()[(index) + 1] = F_gnn[(i*2) + 1]

        # ####### plot ########
        plt.figure()
        self.plot(u.sub(1))
        plt.savefig("Stats/plot_results" + n_epoch + ".jpg")
