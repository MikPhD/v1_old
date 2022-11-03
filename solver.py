from dolfin import *
from logging import getLogger, WARNING


def solve_direct(F, x):
    # Default verbosity for FFC
    getLogger('FFC').setLevel(WARNING)

    comm = MPI.comm_world
    set_log_level(20)  # int=20 -> INFO log_level
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["std_out_all_processes"] = False

    # ----------------PARAMETER--------------------------
    Re = x[0][2]
    nu = Constant(1 / Re)
    # --------------------------------------------------

    #-------------- READ THE INITIAL MESH -----------------------
    mesh_file = "../Dataset/" + str(int(Re)) + "/Mesh.h5"
    mesh = Mesh() #Init mesh class
    with HDF5File(comm, mesh_file, "r") as h5file:
        h5file.read(mesh, "mesh", False)
        facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        h5file.read(facet, "facet")
    # ----------------------------------------------------------

    # ----------- CREATE FEM ELEMENT--------------------
    VelocityElement2 = VectorElement("CG", mesh.ufl_cell(), 2)
    VelocityElement1 = VectorElement("CG", mesh.ufl_cell(), 1)
    PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1)
    Space2 = FunctionSpace(mesh, VelocityElement2 * PressureElement)
    Space1 = FunctionSpace(mesh, VelocityElement1 * PressureElement)


    w = Function(Space2)
    u, p = split(w)

    w_truth = Function(Space2)

    f1 = Function(Space1)

    f1, _ = f1.split(deepcopy=True)

    v, q = TestFunctions(Space2)
    # ---------------------------------------------------

    #----------- project the prediction ------------------------
    F = F.cpu().detach().numpy().flatten()

    mesh_points = mesh.coordinates().tolist()

    dofs_coordinates_prev = Space1.sub(0).collapse().tabulate_dof_coordinates().tolist()
    # try to first create the list and then assign the entire list to f
    for i, z in enumerate(mesh_points):
        index = dofs_coordinates_prev.index(z)
        f1.vector()[(index)] = F[i * 2]
        f1.vector()[(index) + 1] = F[(i * 2) + 1]
    # ----------------------------------------------------------

    f1 = project(f1, Space2.sub(0).collapse())

    #--------- compute direct solver step ----------------------
    # ------------computing the base flow (re<re_cr) --------------
    bcs = []
    bcs.append(DirichletBC(Space2.sub(0), Constant((1.0, 0.0)), facet, 1))
    bcs.append(DirichletBC(Space2.sub(0), Constant((0.0, 0.0)), facet, 2))
    bcs.append(DirichletBC(Space2.sub(0).sub(1), Constant(0.0), facet, 3))
    bcs.append(DirichletBC(Space2.sub(1), Constant(0.0), facet, 4))

    f_base = Constant((0.0, 0.0))

    G = (nu * inner(grad(u), grad(v))
         + dot(dot(grad(u), u), v)
         - p * div(v)
         - q * div(u)
         - dot(f_base, v)) * dx

    J = derivative(G, w)
    problem = NonlinearVariationalProblem(G, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    # ---------------------------------------------------------

    #-------------Reading the truth mean flow -----------------
    with HDF5File(MPI.comm_world, "../Dataset/" + str(int(Re)) + "/Results.h5", "r") as h5file:
        h5file.read(w_truth, "mean")
    #----------------------------------------------------------
    diff = project(w_truth - w, Space2)
    loss = norm(diff)

    return loss