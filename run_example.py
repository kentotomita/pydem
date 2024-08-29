
# This script demonstrates the use of the PyDEM library to generate a synthetic terrain, apply the ALHAT hazard detection algorithm, and visualize the results using PyVista.

# Required Libraries
import numpy as np
import pyvista as pv

import pydem.SyntheticTerrain as st
import pydem.Geometry as gm
import pydem.Graphic as pyvis
import pydem.HazardDetection as hd
import pydem.util as util

# 1. Construct a Sample DEM (Digital Elevation Map)
dem = st.rocky_terrain(shape=(128, 64), res=0.1, k=0.4, dmax=2., dmin=0.2)

# 2. Apply ALHAT Hazard Detection Algorithm
dl = 3.0  # Lander diameter (meters)
dp = 0.3  # Landing pad diameter (meters)
rmpp = 0.1  # Resolution (meters per pixel)

fpmap, site_slope, site_prsafe, pix_prsafe, psafe, indef = hd.alhat(
    dem, 
    rmpp=rmpp, 
    lander_type='square', 
    dl=dl, 
    dp=dp,
    scrit=10*np.pi/180,  # Critical slope (radians)
    rcrit=0.3,  # Critical roughness
    sigma=0.05/3  # Slope estimation error
)

# 3. Create PyVista Mesh for Visualization
h, w = dem.shape
x = np.linspace(0, 12.8, h)
y = np.linspace(0, 6.4, w)
yy, xx = np.meshgrid(y, x)
mesh = pv.StructuredGrid(xx, yy, dem)

# 4. Add Hazard Detection Results as Scalar Attributes
mesh.point_data.set_array(fpmap.flatten('F'), 'fpmap')
mesh.point_data.set_array(site_slope.flatten('F'), 'site_slope')
mesh.point_data.set_array(site_prsafe.flatten('F'), 'site_prsafe')
mesh.point_data.set_array(pix_prsafe.flatten('F'), 'pix_prsafe')
mesh.point_data.set_array(psafe.flatten('F'), 'site_psafe')

# 5. Visualize Results (Example: Pixel-wise Safety due to Roughness)
mesh.set_active_scalars('pix_prsafe')
mesh.plot()

