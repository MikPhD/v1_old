import numpy as np
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import argparse
import traceback
from pdb import set_trace

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def plot(obj):
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        plt.triplot(mesh2triang(obj), color='k')

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch_number', help='gnn epoch number', type=str, default="")
parser.add_argument('-re', '--re_number', help='Reynolds number ', type=str, default="110")
args = parser.parse_args()

epoch_number = args.epoch_number
re_number = args.re_number

directory = "../Dataset/" + re_number

####### loading mesh ########
mesh = Mesh()
with HDF5File(MPI.comm_world, directory + "/Mesh.h5", "r") as h5file:
    h5file.read(mesh, "mesh", False)
    facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    h5file.read(facet, "facet")

####### initializing holder ########
VelocityElement2 = VectorElement("CG", mesh.ufl_cell(), 2)
Space2 = FunctionSpace(mesh, VelocityElement2)

VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
Space = FunctionSpace(mesh, VelocityElement)

F_dns = Function(Space2)
F_gnn = Function(Space)
F_tot = Function(Space)
#
# ####### loading forcing from GNN ################
f_gnn_npy = np.load('./Results/results' + epoch_number + '.npy').flatten()

d2v = dof_to_vertex_map(Space)
a_new = [f_gnn_npy[d2v[i]] for i in range(2*mesh.num_vertices())]

F_gnn.vector().set_local(a_new)
#
######### loading DNS forcing #####################
with HDF5File(MPI.comm_world, directory + "/Results.h5", "r") as h5file:
    h5file.read(F_dns, "forcing")

F_dns = project(F_dns, Space)
#
######### executing difference ####################
F_tot.vector()[:] = F_gnn.vector()[:] - F_dns.vector()[:]
print(norm(F_tot), "norma di F_tot")
print(norm(F_tot)/norm(F_dns), "norma di F_tot normalizzata rispetto a F_dns")
print(norm(F_gnn), "norma di F_gnn")
print(norm(F_dns), "norma di F_dns")
#
######### plot results ##############
plt.figure()
plot(F_tot.sub(0))
plt.show()
