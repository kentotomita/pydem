import numpy as np
import sys
sys.path.append('../../')

import pyvista as pv


import pydem.SyntheticTerrain as st

if __name__=="__main__":
    # make mesh
    dem = st.rocky_terrain(shape=(128, 64), res=0.1, k=0.4, dmax=2., dmin=0.2)
    h, w = dem.shape
    x = np.linspace(0, 12.8, h)
    y = np.linspace(0, 6.4, w)
    yy, xx = np.meshgrid(y, x)
    mesh = pv.StructuredGrid(xx, yy, dem)
    mesh.point_data.set_array(dem.flatten('F'), 'elevation')
    mesh.set_active_scalars('elevation')
    mesh
    mesh.plot()
