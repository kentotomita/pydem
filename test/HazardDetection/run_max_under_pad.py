
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm
import pydem.Graphic as pyvis
import pyvista as pv
from pydem.HazardDetection import _util as hd

if __name__=="__main__":

    #dem = st.gen_crater(d=50, res=1)
    #dem = st.dsa(np.random.random(size=(128, 128)))
    dem = st.rocky_terrain(shape=(128, 64), res=0.1, k=0.4, dmax=2., dmin=0.2)

    h, w = dem.shape
    x = np.linspace(0, 12.8, h)
    y = np.linspace(0, 6.4, w)
    yy, xx = np.meshgrid(y, x)

    mesh = gm.dem2mesh_tri(xx=xx, yy=yy, zz=dem)
    mesh = pv.wrap(mesh)

    fpmap = hd.max_under_pad(n_rpad=5, dem=dem)

    pyvis.pvplot_dem(mesh, fpmap)
