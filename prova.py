import pyvista as pv
from pyvista import examples
import numpy as np

mesh_points = np.load('./mesh_points.npy')
cloud = np.array(np.insert(mesh_points, -1, 0, axis=1))

point_cloud = pv.PolyData(cloud)
mesh = point_cloud.delaunay_2d()

mesh.plot(show_edges=True, line_width=5)
p1 = point_cloud.find_closest_point((0,0,0))
p2 = point_cloud.find_closest_point((30,0,0))
point_cloud.geodesic(p1,p2)
# mesh.plot(eye_dome_lighting=True)
# # a = dolfin.geodesic(p1,p2)