import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import sys
sys.path.append('../')

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm

if __name__=="__main__":

    #dem = st.gen_crater(d=50, res=1)
    #dem = st.dsa(np.random.random(size=(128, 128)))
    dem = st.rocky_terrain(shape=(128, 128), res=0.1, k=0.4, dmax=2., dmin=0.2)

    plt.imshow(dem)
    plt.show()

    w, h = dem.shape
    x = np.linspace(0, 50, w)
    y = np.linspace(0, 50, h)
    xx, yy = np.meshgrid(x, y)

    mesh = gm.dem2mesh_o3d(xx=xx, yy=yy, zz=dem, cuda=False)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
