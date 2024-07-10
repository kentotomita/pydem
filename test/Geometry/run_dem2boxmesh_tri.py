import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm

if __name__=="__main__":

    #dem = st.gen_crater(d=50, res=1)
    #dem = st.dsa(np.random.random(size=(128, 128)))
    dem = st.rocky_terrain(shape=(128, 64), res=0.1, k=0.4, dmax=2.0, dmin=0.2, debug=True)

    plt.imshow(dem)
    plt.show()

    h, w = dem.shape
    x = np.linspace(0, 12.8, h)
    y = np.linspace(0, 6.4, w)
    yy, xx = np.meshgrid(y, x)

    mesh = gm.dem2boxmesh_tri(xx=xx, yy=yy, zz=dem)
    mesh.show()