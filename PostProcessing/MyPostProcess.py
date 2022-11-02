from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import matplotlib as mpl
from collections import OrderedDict
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter


class PostProcess():
    def __init__(self):
        self.cmaps = OrderedDict()
        self.cmaps['Perceptually Uniform Sequential'] = ['plasma_r']

    def mesh2triang(self, mesh):
        xy = mesh.coordinates()
        return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

    def plot(self, obj, label):
        # plt.gca().set_aspect('equal')
        mesh = obj.function_space().mesh()

        if isinstance(obj, Function):
            if obj.vector().size() == mesh.num_cells():
                C = obj.vector().array()
                plt.tripcolor(self.mesh2triang(mesh), C, cmap=self.cmaps, norm='Normalize')

            else:
                x = mesh.coordinates()[:, 0]
                y = mesh.coordinates()[:, 1]

                #trick to center the axes wrt the cilinder
                if label == "gnn":
                    x -= 11.25
                    y -= 5

                t = mesh.cells()
                v = obj.compute_vertex_values(mesh)
                vmin = v.min()
                vmax = v.max()

                print(f'vmax {vmax}, vmin {vmin}')

                # vmin = -0.2287
                # vmax = 0.3112

                # if label == 'gnn':
                #     vmin = -0.6
                #     vmax = 0.4
                # else:
                #     vmin = -0.2
                #     vmax = 0.1
                #
                # vmin = -0.01
                # vmax = 0.01

                v[v < vmin] = vmin + 1e-12
                v[v > vmax] = vmax - 1e-12
                from matplotlib.ticker import ScalarFormatter
                cmap = 'viridis'
                levels = np.linspace(vmin, vmax, 100)
                formatter = ScalarFormatter()
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                c = ax.tricontourf(x, y, t, v, levels=levels, norm=norm,
                                   cmap=plt.get_cmap(cmap))
                plt.axis('equal')
                p = ax.triplot(x, y, t, '-', lw=0.5, alpha=0.0)
                ax.set_xlim([x.min(), x.max()])
                ax.set_ylim([y.min(), y.max()])
                ax.set_xlabel(' $\it{\:x}$')
                ax.set_ylabel(' $\it{\:y}$')
                # tit = plt.title('Componente x del tensore di Reynolds')
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes('right', "4%", pad="2%")

                colorbar_format = '% 1.3f'
                cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, format=colorbar_format)
                fig.savefig("./cases/" + case_name + f"/plot_{label}.png")


        elif isinstance(obj, Mesh):
            plt.triplot(self.mesh2triang(obj), color='k')

    def plot_results(self):
        ####### loading mesh ########
        mesh = Mesh()
        mesh_file = "./Mesh.h5"
        with HDF5File(MPI.comm_world, mesh_file, "r") as h5file:
            h5file.read(mesh, "mesh", False)
            facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
            h5file.read(facet, "facet")

        ####### initializing holder ########
        VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
        PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
        Space = FunctionSpace(mesh, VelocityElement * PressureElement)

        F = Function(Space)
        v, f = F.split(deepcopy=True)

        # with HDF5File(MPI.comm_world, "../Dataset/" + case_num + "/Results.h5", "r") as h5file:
        #     h5file.read(f, "mean")
        #     h5file.read(f, "forcing")

        # ####### loading forcing from GNN ################
        F_gnn = np.load("./results.npy").flatten()
        mesh_points = mesh.coordinates().tolist()

        dofs_coordinates_prev = Space.sub(0).collapse().tabulate_dof_coordinates().tolist()

        for i, x in enumerate(mesh_points):
            index = dofs_coordinates_prev.index(x)
            v.vector()[(index)] = F_gnn[i * 2]
            v.vector()[(index) + 1] = F_gnn[(i * 2) + 1]

        ######### plot results ##############
        plt.figure()
        self.plot(v.sub(0))
        plt.show()

    def differences(self, case_name):
        ####### loading mesh ########
        mesh = Mesh()
        mesh_file = "./cases/" + case_name + "/Mesh.h5"
        with HDF5File(MPI.comm_world, mesh_file, "r") as h5file:
            h5file.read(mesh, "mesh", False)
            facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
            h5file.read(facet, "facet")

        ####### initializing holder ########
        VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
        PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
        Space = FunctionSpace(mesh, VelocityElement * PressureElement)

        F_gnn = Function(Space)
        v_gnn, f_gnn = F_gnn.split(deepcopy=True)

        F_cfd = Function(Space)
        v_cfd, f_cfd = F_cfd.split(deepcopy=True)

        F_diff = Function(Space)
        v_diff, f_diff = F_diff.split(deepcopy=True)


        ######## loading forcing from GNN ################
        F_gnn = np.load("./cases/" + case_name + "/F_gnn.npy").flatten()

        ######## loading forcing from CFD ################
        F_cfd = np.load("./cases/" + case_name + "/F_dns.npy").flatten()

        ######## mesh coordinates ##############
        mesh_points = mesh.coordinates().tolist()

        dofs_coordinates_prev = Space.sub(0).collapse().tabulate_dof_coordinates().tolist()

        for i, x in enumerate(mesh_points):
            index = dofs_coordinates_prev.index(x)
            v_gnn.vector()[(index)] = F_gnn[i * 2]
            v_gnn.vector()[(index) + 1] = F_gnn[(i * 2) + 1]

        for i, x in enumerate(mesh_points):
            index = dofs_coordinates_prev.index(x)
            v_cfd.vector()[(index)] = F_cfd[i * 2]
            v_cfd.vector()[(index) + 1] = F_cfd[(i * 2) + 1]

        v_diff.vector()[:] = v_cfd.vector()[:] - v_gnn.vector()[:]

        ######### plot results ##############
        self.plot(v_gnn.sub(0), "gnn")
        #
        self.plot(v_cfd.sub(0), "dns")

        self.plot(v_diff.sub(0), "diff")

        print(f"Norma della differenza delle funzioni: {norm(v_diff, 'L2')}")
        print(f"Norma della differenza delle funzioni normalizzata: {norm(v_diff, 'L2')/norm(v_cfd, 'L2')}")


case_name = input('Insert case name: [downshift, re130, re200, very_down, down_confined, flipflop_re60g1, '
                  'cavity, tiltsquare, tiltsquare_smooth, upstream, flipflop_re55g1_5, flipflop_re60g1_5, '
                  're260, re100, re110, square, re40, re50, re180]')
post_process = PostProcess()
# post_process.plot_results()
post_process.differences(case_name)

