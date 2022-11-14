from dolfin import *
from logging import getLogger, WARNING
from MyPlot import Plot
import numpy as np

def solve_direct(F_gnn, Re = 150):
    # Default verbosity for FFC
    getLogger('FFC').setLevel(WARNING)

    comm = MPI.comm_world
    set_log_level(20)  # int=20 -> INFO log_level
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["std_out_all_processes"] = False

    #-------Project NN result to FEniCS------
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
    Space1 = FunctionSpace(mesh, VelocityElement * PressureElement)

    F1 = Function(Space1)

    f1, _ = F1.split(deepcopy=True)

    # ####### loading forcing from GNN ################
    # F_gnn = np.load("./Results/F.npy").flatten()

    F_gnn = F_gnn.flatten()

    # mesh_points = np.load('./Results/mesh_points.npy').tolist()
    mesh_points = mesh.coordinates().tolist()

    dofs_coordinates_prev = Space1.sub(0).collapse().tabulate_dof_coordinates().tolist()

    for i, x in enumerate(mesh_points):
        index = dofs_coordinates_prev.index(x)
        f1.vector()[(index)] = F_gnn[i * 2]
        f1.vector()[(index) + 1] = F_gnn[(i * 2) + 1]

    # ----------------PARAMETER--------------------------
    nu = Constant(1 / Re)
    # --------------------------------------------------

    # ----------- CREATE FEM ELEMENT--------------------
    VelocityElement = VectorElement("CG", mesh.ufl_cell(), 2)
    PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
    Space = FunctionSpace(mesh, VelocityElement * PressureElement)

    w = Function(Space)
    u, p = split(w)
    F = Function(Space)
    f, _ = split(F)

    v, q = TestFunctions(Space)
    # ---------------------------------------------------

    # f = project(f, Space.sub(0).collapse())

    # with HDF5File(MPI.comm_world, "./Results/Results.h5", "r") as h5file:
    #     h5file.read(f, "forcing")

    # ----------- P1 to P2 projection --------------------
    f = project(f1, Space.sub(0).collapse())
    # ---------------------------------------------------

    #--------- compute direct solver step ----------------------
    # ------------computing the base flow (re<re_cr) --------------
    bcs = []
    bcs.append(DirichletBC(Space.sub(0), Constant((1.0, 0.0)), facet, 1)) #inflow
    bcs.append(DirichletBC(Space.sub(0), Constant((0.0, 0.0)), facet, 2)) #cylinder
    bcs.append(DirichletBC(Space.sub(0).sub(1), Constant(0.0), facet, 3)) #wall
    bcs.append(DirichletBC(Space.sub(1), Constant(0.0), facet, 4)) #outflow

    T = (+ nu*inner(grad(u), grad(v))

        # + inner(grad(u)*u_prev1, v)
        # + inner(grad(u_prev1)*u, v)
        # + inner(grad(u_prev1)*u_prev1, v)

        + inner(grad(u) * u, v)

        - inner(p, div(v))
        - inner(q, div(u))
        - inner(f, v)
        - Constant(1e-6) * p * q) * dx

    # T = (+nu * inner(grad(u), grad(v))
    #      + inner(grad(u) * u, v)
    #      - inner(p, div(v))
    #      - inner(q, div(u))
    #      - inner(f, v)
    #      - Constant(1e-6) * p * q
    #      ) * dx

    J = derivative(T, w)
    problem = NonlinearVariationalProblem(T, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    # ---------------------------------------------------------

    #----------------- setting the cost functional -------------
    # J = norm(u - )

    # #----------- project the prediction ------------------------
    # F = F.cpu().detach().numpy().flatten()
    #
    # mesh_points = mesh.coordinates().tolist()
    #
    # dofs_coordinates_prev = Space1.sub(0).collapse().tabulate_dof_coordinates().tolist()
    # # try to first create the list and then assign the entire list to f
    # for i, z in enumerate(mesh_points):
    #     index = dofs_coordinates_prev.index(z)
    #     f1.vector()[(index)] = F[i * 2]
    #     f1.vector()[(index) + 1] = F[(i * 2) + 1]
    # # ----------------------------------------------------------

    # f1 = project(f1, Space2.sub(0).collapse())

    return w

if __name__ == "__main__":
    my_plot = Plot()

    w = solve_direct(Re = 50)

    my_plot.plot(w.sub(0).sub(0), 0)
