from MyDSS import MyOwnDSSNet
from Mydataset import MyOwnDataset

from torch_geometric.data import DataLoader
import os
import torch
import numpy as np
from fenics import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


#check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on : ', device)

test_case = ['100']

#clean processed folder (Model comes from an older version of PYG)
if os.path.exists("./dataset/processed/data_val.pt"):
    os.remove("./dataset/processed/data_val.pt")
if os.path.exists("./dataset/processed/data_train.pt"):
    os.remove("./dataset/processed/data_train.pt")
if os.path.exists("./dataset/processed/data_test.pt"):
    os.remove("./dataset/processed/data_test.pt")
if os.path.exists("./dataset/processed/pre_filter.pt"):
    os.remove("./dataset/processed/pre_filter.pt")
if os.path.exists("./dataset/processed/pre_transform.pt"):
    os.remove("./dataset/processed/pre_transform.pt")

print("#################### DATA ADAPTING FOR GNN #######################")
################# inizio lettura file ##########################
######### lettura mesh #########
mesh = Mesh()
mesh_file = "./Test/res_mesh/Mesh.h5"
with HDF5File(MPI.comm_world, mesh_file, "r") as h5file:
    h5file.read(mesh, "mesh", False)
    facet = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    h5file.read(facet, "facet")

###### lettura risultati ########
VelocityElement2 = VectorElement("CG", mesh.ufl_cell(), 2)
PressureElement2 = FiniteElement("CG", mesh.ufl_cell(), 1)
Space2 = FunctionSpace(mesh, VelocityElement2 * PressureElement2)

u_glob2 = Function(Space2)
f_glob2 = Function(Space2)

with HDF5File(MPI.comm_world, "./Test/res_mesh/Results.h5", "r") as h5file:
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

C = []  ##connection list
D = []  ##distances between connection
for i in range(mesh.num_edges()):
    connection = np.array(mesh_connectivity(i)).astype(int)
    coord_vert1 = (mesh.coordinates()[connection[0]]).tolist()
    coord_vert2 = (mesh.coordinates()[connection[1]]).tolist()
    if coord_vert1 and coord_vert2 in bmesh:
        pass
    elif coord_vert1 in bmesh:
        distancex = coord_vert2[0] - coord_vert1[0]
        distancey = coord_vert2[1] - coord_vert1[1]
        C.append(list(connection))
        D.append([distancex, distancey])
    elif coord_vert2 in bmesh:
        pass
    else:
        connection_rev = connection[::-1]
        distancex = coord_vert2[0] - coord_vert1[0]
        distancey = coord_vert2[1] - coord_vert1[1]

        C.append(list(connection))
        C.append(list(connection_rev))

        D.append([distancex, distancey])
        D.append([-1 * distancex, -1 * distancey])

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
os.makedirs("./dataset/raw/test/" + test_case[0], exist_ok=True)
specific_dir = "./dataset/raw/test/" + test_case[0]
np.save(specific_dir + "/C.npy", C)
np.save(specific_dir + "/D.npy", D)
np.save(specific_dir + "/U.npy", U)
np.save(specific_dir + "/F.npy", F)
np.save(specific_dir + "/re.npy", int(test_case[0]))
################# Fine salvataggio file ##################################

################# Print interface ########################################
print("Trasformazione file di completata!")

print("#################### CREATING Inner DATASET #######################")
loader_test = MyOwnDataset(root='./dataset', mode='test', cases=test_case, device=device)
#initialize the created dataset
loader_test = DataLoader(loader_test)

print("#################### DSS NET parameter #######################")
#create hyperparameter
latent_dimension = 18
print("Latent space dim : ", latent_dimension)
k = 87
print("Number of updates : ", k)
gamma = 0.1
print("Gamma (loss function) : ", gamma)
alpha = 1e-2
print("Alpha (reduction correction) :", alpha)
lr = 3e-3
print("LR (Learning rate):", lr)

##create folder for different results ##
set_name = str(k) + '-' + str(latent_dimension).replace(".", "") + '-' + str(alpha).replace(".", "") + '-' + str(
    lr).replace(".", "")
print("PARAMETER SET: k:{}, laten_dim:{}, alpha:{}, lr:{}".format(str(k), str(latent_dimension), str(alpha), str(lr)))
os.makedirs("./Test/" + set_name, exist_ok=True)

print("#################### CREATING NETWORKS #######################")
DSS = MyOwnDSSNet(latent_dimension=latent_dimension, k=k, gamma=gamma, alpha=alpha, device=device)
DSS = DSS.to(device)

print("#################### TESTING #######################")
optimizer = torch.optim.Adam(DSS.parameters(), lr=lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=50,
                                                       min_lr=0.001, verbose=True)

#Load checkpoint
checkpoint = torch.load('Model/best_model.pt', map_location=torch.device('cpu'))
DSS.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
min_val_loss = checkpoint['min_val_loss']

DSS.eval()

with torch.no_grad():
    for test_data in loader_test:
        F, test_loss, loss_dict = DSS(test_data)
        total_test_loss = test_loss.sum().item()
        final_loss_test = loss_dict[str(k)].sum().item()

print("Test loss = {:.5e}".format(total_test_loss / len(loader_test)))

#detach result for plot
F_gnn = F[str(k)].cpu().numpy()
np.save("./Test/" + set_name + "/F_gnn.npy", F_gnn)

print("#################### PLOT RESULTS #######################")
F_gnn = np.load("./Test/" + set_name + "/F_gnn.npy").flatten()

####### loading mesh ########
mesh = Mesh()
mesh_file = "./Test/res_mesh/Mesh.h5"
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

# ####### loading forcing from GNN ################
F_gnn = F_gnn.flatten()
mesh_points = mesh.coordinates().tolist()

dofs_coordinates_prev = Space.sub(0).collapse().tabulate_dof_coordinates().tolist()

for i, x in enumerate(mesh_points):
    index = dofs_coordinates_prev.index(x)
    u.vector()[(index)] = F_gnn[i * 2]
    u.vector()[(index) + 1] = F_gnn[(i * 2) + 1]

######## plot ########
x = mesh.coordinates()[:, 0]
y = mesh.coordinates()[:, 1]
t = mesh.cells()
v = u.sub(0).compute_vertex_values(mesh)
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

fig_plot.savefig("./Test/" + set_name + "/plot_results.jpg")
plt.close(fig_plot)










