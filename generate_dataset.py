# generate_dataset.py
# HOW TO USE:
# python generate_dataset.py

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
import sys
sys.path.append("./") # This is to access the pydem module

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm
import pydem.Graphic as pyvis
import pydem.HazardDetection as hd
import pydem.util as util


# generate a rocky terrain as an example
nh, nw = 512, 512 # height and width of the terrain
res = 0.1 # resolution of the terrain

# generate a rocky terrain
dem = st.rocky_terrain(shape=(nh, nw), res=res, k=0.3, dmax=1.5, dmin=0.1) # k=0.3 means 30% of the terrain is covered by rocks

# generate a crater? #FIXME: what is this?
terrain = st.dsa(dem, hmax=0.7) 

# smooth the terrain
from scipy.ndimage import gaussian_filter
dem += gaussian_filter(terrain, sigma=2)


dl = 3.0 #FIXME: what is this?
dp = 0.3 #FIXME: what is this?
rmpp = 0.1 #FIXME: what is this?


fpmap, site_slope, site_rghns, pix_rghns, site_safe, indef = hd.dhd(
    dem, 
    rmpp=rmpp, 
    negative_rghns_unsafe=False,
    lander_type='square', 
    dl=dl, 
    dp=dp,
    scrit=10*np.pi/180,
    rcrit=0.3
)

# Create mesh from DEM
h, w = dem.shape
x = np.linspace(0, nh * res, h)
y = np.linspace(0, nw * res, w)
xc, yc = np.mean(x), np.mean(y)
x -= xc
y -= yc
yy, xx = np.meshgrid(y, x)
mesh = pv.StructuredGrid(xx, yy, dem)

# Add scalar attributes to mesh
mesh.point_data.set_array(fpmap.flatten('F'), 'fpmap')
mesh.point_data.set_array(site_slope.flatten('F'), 'site_slope')
mesh.point_data.set_array(site_rghns.flatten('F'), 'site_rghns')
mesh.point_data.set_array(pix_rghns.flatten('F'), 'pix_rghns')
mesh.point_data.set_array(site_safe.flatten('F'), 'site_safe')

mesh.set_active_scalars('site_safe')
# mesh.plot()
plt.imshow(site_safe, cmap='gray')
plt.show()
