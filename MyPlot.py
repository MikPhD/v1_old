import numpy as np
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import traceback
import argparse
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
parser.add_argument('-i', '--n_epoch', help='epoch number', type=str, default="")
args = parser.parse_args()

n_epoch = args.n_epoch

directory = "./Results/"

####### loading mesh ########
mesh = Mesh()
mesh_file = directory + "Mesh.h5"
with HDF5File(MPI.comm_world, mesh_file, "r") as h5file:
    h5file.read(mesh, "mesh", False)
    facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    h5file.read(facet, "facet")

####### initializing holder ########
VelocityElement = VectorElement("CG", mesh.ufl_cell(), 1)
Space = FunctionSpace(mesh, VelocityElement)

F = Function(Space)
# F.interpolate(Constant((0.0, 0.0)))

# ####### loading forcing from GNN ################
F_gnn = np.load(directory + 'results' + n_epoch + '.npy').flatten()
# coord = np.load(directory + 'coord.npy')
#
d2v = dof_to_vertex_map(Space)
a_new = [F_gnn[d2v[i]] for i in range(2*mesh.num_vertices())]

F.vector().set_local(a_new)
#
# ####### plot ########
plt.figure()
plot(F.sub(0))
plt.show()
#
# # print(w.vector()[:])
# # print(U.flatten())
# # print(mesh.num_vertices())
