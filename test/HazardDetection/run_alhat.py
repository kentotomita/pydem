import numpy as np

import sys
sys.path.append('../../')

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm
import pydem.Graphic as pyvis
import pyvista as pv
import pydem.HazardDetection as hd
import pydem.util as util

if __name__=="__main__":

    #dem = st.gen_crater(d=50, res=1)
    #dem = st.dsa(np.random.random(size=(128, 128)))
    dem = st.rocky_terrain(shape=(128, 64), res=0.1, k=0.4, dmax=2., dmin=0.2)

    h, w = dem.shape
    x = np.linspace(0, 12.8, h)
    y = np.linspace(0, 6.4, w)
    yy, xx = np.meshgrid(y, x)

    mesh = pv.StructuredGrid(xx, yy, dem)

    dl = 3.0
    dp = 0.3
    rmpp = 0.1

    fpmap, site_slope, site_prsafe, pix_prsafe, psafe, indef = hd.alhat(
        dem, 
        rmpp=rmpp, 
        lander_type='square', 
        dl=dl, 
        dp=dp,
        scrit=10*np.pi/180,
        rcrit=0.3,
        sigma=0.05/3)
    
    
    # https://docs.pyvista.org/version/stable/user-guide/data_model.html
    mesh.point_data.set_array(fpmap.flatten('F'), 'fpmap')
    mesh.point_data.set_array(site_slope.flatten('F'), 'site_slope')
    mesh.point_data.set_array(site_prsafe.flatten('F'), 'site_prsafe')
    mesh.point_data.set_array(pix_prsafe.flatten('F'), 'pix_prsafe')
    mesh.point_data.set_array(psafe.flatten('F'), 'site_psafe')

    mesh.set_active_scalars('fpmap')
    mesh.plot()

    mesh.set_active_scalars('site_slope')
    mesh.plot()

    mesh.set_active_scalars('site_prsafe')
    mesh.plot()

    mesh.set_active_scalars('pix_prsafe')
    mesh.plot()

    mesh.set_active_scalars('site_psafe')
    mesh.plot()
    
    
    