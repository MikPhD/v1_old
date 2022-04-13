from fenics import *
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import matplotlib.tri as tri


class CreateData:
    def __init__(self):
        self.comm = MPI.comm_world
        set_log_level(40)
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters["std_out_all_processes"] = False

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

    def transform(self, cases, mode):
        for h in cases:
            #####initializing####
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


            ############# CG 2 to CG 1 ##############
            # è necessario perchè lavoro sugli indici dei vertici della mesh!
            VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
            PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
            Space = FunctionSpace(mesh, VelocityElement * PressureElement)

            u_glob = project(u_glob2, Space)
            f_glob = project(f_glob2, Space)
            ################### End lettura file ####################################

            ################### Creazione elementi dataset ##########################
            #### lista delle connessioni della mesh ###
            mesh_points = mesh.coordinates().tolist()
            bmesh = BoundaryMesh(mesh, "exterior", True).coordinates().tolist()

            ### analizzo ogni edge della mesh ###
            mesh.init()
            mesh_topology = mesh.topology()
            mesh_connectivity = mesh_topology(1, 0)

            C = [] ##connection list
            D = [] ##distances between connection
            for i in range(mesh.num_edges()):
                connection = np.array(mesh_connectivity(i)).astype(int)
                coord_vert1 = (mesh.coordinates()[connection[0]]).tolist()
                coord_vert2 = (mesh.coordinates()[connection[1]]).tolist()

                connection_rev = connection[::-1]
                distancex = coord_vert2[0]-coord_vert1[0]
                distancey = coord_vert2[1]-coord_vert1[1]

                C.append(list(connection))
                C.append(list(connection_rev))

                D.append([distancex, distancey])
                D.append([-1 * distancex, -1 * distancey])

                # if coord_vert1 and coord_vert2 in bmesh:
                #     pass
                # elif coord_vert1 in bmesh:
                #     distancex = coord_vert2[0] - coord_vert1[0]
                #     distancey = coord_vert2[1] - coord_vert1[1]
                #     C.append(list(connection))
                #     D.append([distancex, distancey])
                # elif coord_vert2 in bmesh: ##never happen for how fenics order the points in the iteration
                #     pass
                # else:
                #     connection_rev = connection[::-1]
                #     distancex = coord_vert2[0]-coord_vert1[0]
                #     distancey = coord_vert2[1]-coord_vert1[1]
                #
                #     C.append(list(connection))
                #     C.append(list(connection_rev))
                #
                #     D.append([distancex, distancey])
                #     D.append([-1 * distancex, -1 * distancey])



            ############## mean flow e forcing #####################
            ########### mappatura attraverso coordinate ############

            U_P = []
            F = []
            for x in mesh_points:
                U_P.append(list(u_glob(np.array(x))))
                F.append(list(f_glob(np.array(x))))

            ###### remove useless component -> z component#####
            for j in F:
                del j[2]
            ###### remove pressure ########
            for i in U_P:
                del i[2]

            U = U_P

        ######################### Fine creazione elementi############################

        ###################### Salvataggio file Numpy ##############################
            os.makedirs("./dataset/raw/" + mode + "/" + h, exist_ok=True)
            specific_dir = "./dataset/raw/" + mode + "/" + h
            np.save(specific_dir + "/C.npy", C)
            np.save(specific_dir + "/D.npy", D)
            np.save(specific_dir + "/U.npy", U)
            np.save(specific_dir + "/F.npy", F)
            # np.save(specific_dir + "/coord.npy", coord)
            np.save(specific_dir + "/re.npy", int(h))
            # np.save("./mesh_points.npy", mesh_points)
        ################# Fine salvataggio file ##################################

        ################# Print interface ########################################
        print("Trasformazione file di " + mode + " completata!")


        # ### plot control ####
        # plt.figure()
        # plot(mesh)
        # for x, y in mesh_points:
        #     plt.plot(x, y, 'r*')
        # plt.show()
        #
        # plot(mesh)
        # for x, y in bmesh:
        #     plt.plot(x, y, 'b*')
        # plt.show()
        # x = []
        # y = []
        # plot(mesh)
        # for i, (index1, index2) in enumerate(C[0:1000]):
        #     xv1 = mesh_points[index1][0]
        #     yv1 = mesh_points[index1][1]
        #     xv2 = mesh_points[index2][0]
        #     yv2 = mesh_points[index2][1]
        #     xmean = (xv1+xv2)/2
        #     ymean = (yv1+yv2)/2
        #
        #     u = xv2-xv1
        #     v = yv2-yv1
        #
        #     plt.plot([xv1,xv2], [yv1, yv2], 'ro-', label=i)
        #     plt.annotate(i, xy=(xmean, ymean), xycoords='data')
        #     plt.quiver(xmean, ymean, u, v)
        # #
        # plt.show()

