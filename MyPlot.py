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


class Plot:
    def __init__(self, set_name=""):
        self.set_name = set_name
        self.cmaps = OrderedDict()
        self.cmaps['Perceptually Uniform Sequential'] = ['plasma_r']

    def plot_loss(self):
        ### Open log files
        with open('Stats/' + self.set_name + '/loss_train_log.txt', 'r') as f_train:
            mydata_train = ast.literal_eval(f_train.read())
        with open('Stats/' + self.set_name + '/loss_val_log.txt', 'r') as f_val:
            mydata_val = ast.literal_eval(f_val.read())

        ### define axis and data ###
        dt = 1
        x = np.arange(0, len(mydata_train), dt)
        y_train = mydata_train
        y_val = mydata_val

        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        ax_loss.semilogy(x, y_train, label='train')
        ax_loss.semilogy(x, y_val, label='val')
        ax_loss.legend()
        fig_loss.savefig("Stats/" + self.set_name + "plot_loss.jpg")
        fig_loss.savefig("Stats/" + self.set_name + "/plot_loss.jpg")
        plt.close(fig=fig_loss)


        ### Close Files ###
        f_train.close()
        f_val.close()

    def mesh2triang(self, mesh):
        xy = mesh.coordinates()
        return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    def plot(self, obj, n_epoch):
        # plt.gca().set_aspect('equal')
        mesh = obj.function_space().mesh()

        if isinstance(obj, Function):
            if obj.vector().size() == mesh.num_cells():
                C = obj.vector().array()
                plt.tripcolor(self.mesh2triang(mesh), C, cmap=self.cmaps, norm='Normalize')

            else:
                x = mesh.coordinates()[:, 0]
                y = mesh.coordinates()[:, 1]
                t = mesh.cells()
                v = obj.compute_vertex_values(mesh)
                vmin = v.min()
                vmax = v.max()
                v[v < vmin] = vmin + 1e-12
                v[v > vmax] = vmax - 1e-12
                from matplotlib.ticker import ScalarFormatter
                cmap = 'viridis'
                levels = np.linspace(vmin, vmax, 100)
                formatter = ScalarFormatter()
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                fig_plot, ax_plot = plt.subplots(figsize=(10, 5))
                c = ax_plot.tricontourf(x, y, t, v, levels=levels, norm=norm,
                                   cmap=plt.get_cmap(cmap))
                ax_plot.axis('equal')
                # plt.axis('equal')
                p = ax_plot.triplot(x, y, t, '-', lw=0.5, alpha=0.0)
                ax_plot.set_xlim([x.min(), x.max()])
                ax_plot.set_ylim([y.min(), y.max()])
                ax_plot.set_xlabel(' $\it{Coordinata\:x}$')
                ax_plot.set_ylabel(' $\it{Coordinata\:y}$')
                # tit = plt.title('Componente x del tensore di Reynolds')
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes('right', "4%", pad="2%")
                colorbar_format = '% 1.1f'
                cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, format=colorbar_format)

                fig_plot.savefig("Stats/" + self.set_name + "plot_results.jpg")
                fig_plot.savefig("Stats/" + self.set_name + "/plot_results" + str(n_epoch) + ".jpg")
                plt.close(fig_plot)


        elif isinstance(obj, Mesh):
            plt.triplot(self.mesh2triang(obj), color='k')

    def plot_results(self, n_epoch):

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
        f.set_allow_extrapolation(True)

        # with HDF5File(MPI.comm_world, "../Dataset/" + case_num + "/Results.h5", "r") as h5file:
        #     h5file.read(f, "mean")
        #     h5file.read(f, "forcing")

        # ####### loading forcing from GNN ################
        F_gnn = np.load("./Results/" + self.set_name + "/results" + str(n_epoch) + ".npy").flatten()
        # mesh_points = np.load('./Results/mesh_points.npy').tolist()
        mesh_points = mesh.coordinates().tolist()

        dofs_coordinates_prev = Space.sub(0).collapse().tabulate_dof_coordinates().tolist()

        for i, x in enumerate(mesh_points):
            index = dofs_coordinates_prev.index(x)
            u.vector()[(index)] = F_gnn[i*2]
            u.vector()[(index) + 1] = F_gnn[(i*2) + 1]

        # ####### plot ########
        self.plot(u.sub(0), n_epoch)

