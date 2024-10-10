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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm # Import tqdm for progress bar

nh, nw = 512, 512 # height and width of the dem
res = 0.1        # resolution of the dem

dl = 3.0
dp = 0.3
rmpp = 0.1

# Genearate random dems and save both the dem and site_safe labels
n_training = 1500   # number of training samples
n_validation = 750 # number of validation samples
n_test = 750        # number of test samples

# create training dem directory
import os
os.makedirs('./data/training/depth_maps', exist_ok=True)
os.makedirs('./data/training/label/is_safe', exist_ok=True)
os.makedirs('./data/training/images', exist_ok=True) # Create directory for png images
os.makedirs('./data/validation/depth_maps', exist_ok=True)
os.makedirs('./data/validation/label/is_safe', exist_ok=True)
os.makedirs('./data/validation/images', exist_ok=True) # Create directory for png images
os.makedirs('./data/test/depth_maps', exist_ok=True)
os.makedirs('./data/test/label/is_safe', exist_ok=True)
os.makedirs('./data/test/images', exist_ok=True) # Create directory for png images

for i in tqdm(range(n_training), desc="Generating data"):
    # --- Randomization Enhancements ---
    k = np.random.uniform(0.1, 0.5)  # Vary the k parameter
    dmax = np.random.uniform(1.0, 2.0)  # Vary the dmax parameter
    dmin = np.random.uniform(0.05, 0.2)  # Vary the dmin parameter

    dem = st.rocky_terrain(shape=(nh, nw), res=res, k=0.1, dmax=1.5, dmin=0.1)
    terrain = st.dsa(dem, hmax=0.7)

    
    dem_train = dem + gaussian_filter(terrain, sigma=3)
    depth_data = dem_train.copy()
    x, y = np.mgrid[0:depth_data.shape[0], 0:depth_data.shape[1]]
    fpmap, site_slope, site_rghns, pix_rghns, site_safe, indef = hd.dhd(
        dem_train, 
        rmpp=rmpp, 
        negative_rghns_unsafe=False,
        lander_type='square', 
        dl=dl, 
        dp=dp,
        scrit=10*np.pi/180,
        rcrit=0.3)
    
    # save 8digit file name
    dem_file = f'./data/training/depth_maps/{i:08d}.npy'
    site_safe_file = f'./data/training/label/is_safe/{i:08d}.npy'
    np.save(dem_file, dem_train)
    np.save(site_safe_file, site_safe)

    # Save png image
    plt.figure(figsize=(8, 8))
    plt.imshow(dem_train, cmap='viridis')
    plt.axis('off')
    plt.savefig(f'./data/training/images/{i:08d}.png')
    plt.close()

    # log 
    # print(f"Training: {i}/{n_training}")
    
    if i <= n_validation:
        dem_valid = dem + gaussian_filter(terrain, sigma=3)
        depth_data = dem_valid.copy()
        x, y = np.mgrid[0:depth_data.shape[0], 0:depth_data.shape[1]]
        fpmap, site_slope, site_rghns, pix_rghns, site_safe, indef = hd.dhd(
            dem_valid, 
            rmpp=rmpp, 
            negative_rghns_unsafe=False,
            lander_type='square', 
            dl=dl, 
            dp=dp,
            scrit=10*np.pi/180,
            rcrit=0.3)
        
        # save 8digit file name
        dem_file = f'./data/validation/depth_maps/{i:08d}.npy'
        site_safe_file = f'./data/validation/label/is_safe/{i:08d}.npy'
        np.save(dem_file, dem_valid)
        np.save(site_safe_file, site_safe)

        # Save png image
        plt.figure(figsize=(8, 8))
        plt.imshow(dem_valid, cmap='viridis')
        plt.axis('off')
        plt.savefig(f'./data/validation/images/{i:08d}.png')
        plt.close()
    
        # print(f"Validation: {i}/{n_validation}")

    if i <= n_test:
        dem_test = dem + gaussian_filter(terrain, sigma=3)
        depth_data = dem_test.copy()
        x, y = np.mgrid[0:depth_data.shape[0], 0:depth_data.shape[1]]
        fpmap, site_slope, site_rghns, pix_rghns, site_safe, indef = hd.dhd(
            dem_test, 
            rmpp=rmpp, 
            negative_rghns_unsafe=False,
            lander_type='square', 
            dl=dl, 
            dp=dp,
            scrit=10*np.pi/180,
            rcrit=0.3)
        
        # save 8digit file name
        dem_file = f'./data/test/depth_maps/{i:08d}.npy'
        site_safe_file = f'./data/test/label/is_safe/{i:08d}.npy'
        np.save(dem_file, dem_test)
        np.save(site_safe_file, site_safe)

        # Save png image
        plt.figure(figsize=(8, 8))
        plt.imshow(dem_test, cmap='viridis')
        plt.axis('off')
        plt.savefig(f'./data/test/images/{i:08d}.png')
        plt.close()

        # print(f"Test: {i}/{n_test}")


print("Done!")

