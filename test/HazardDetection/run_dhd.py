import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm
import pydem.Graphic as pyvis
import pyvista as pv
import pydem.HazardDetection as hd

if __name__=="__main__":

    #dem = st.gen_crater(d=50, res=1)
    dem = st.dsa(np.random.random(size=(128, 128)))
    #dem = st.rocky_terrain(shape=(128, 64), res=0.1, k=0.4, dmax=2., dmin=0.2)

    h, w = dem.shape
    x = np.linspace(0, 12.8, h)
    y = np.linspace(0, 12.8, w)
    yy, xx = np.meshgrid(y, x)

    mesh = pv.StructuredGrid(xx, yy, dem)

    fpmap, site_slope, site_rghns, pix_rghns, is_safe, indef = hd.dhd(
        dem, 
        rmpp=0.1, 
        negative_rghns_unsafe=False,
        lander_type='square', 
        dl=3.0, 
        dp=0.3,
        scrit=10*np.pi/180,
        rcrit=0.3)
    
    # https://docs.pyvista.org/version/stable/user-guide/data_model.html
    mesh.point_data.set_array(fpmap.flatten('F'), 'fpmap')
    mesh.point_data.set_array(site_slope.flatten('F'), 'site_slope')
    mesh.point_data.set_array(site_rghns.flatten('F'), 'site_rghns')
    mesh.point_data.set_array(pix_rghns.flatten('F'), 'pix_rghns')
    mesh.point_data.set_array(is_safe.flatten('F'), 'is_safe')

    mesh.set_active_scalars('fpmap')
    mesh.plot()

    mesh.set_active_scalars('site_slope')
    mesh.plot()

    mesh.set_active_scalars('site_rghns')
    mesh.plot()

    mesh.set_active_scalars('pix_rghns')
    mesh.plot()

    mesh.set_active_scalars('is_safe')
    mesh.plot()
    
    