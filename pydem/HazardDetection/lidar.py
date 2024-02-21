"""LiDAR sensor simulation"""

import numpy as np
from trimesh import Trimesh


def define_rays_c(dsize: int=1024, fov_width: float=100.0, alt: float=500.0, pattern: str='grid'):
    """
    Generate rays in camera frame

    Args:
        fov_width (float, optional): Field of view width. Defaults to 100.0.
        alt (float, optional): Altitude. Defaults to 500.0.
        dsize (int, optional): Resolution parameter. Number of data points within 100 m at altitude of 500 m. Defaults to 1024.
        pattern (str, optional): Pattern of rays. Defaults to 'grid'.

    Returns:
        np.ndarray: Rays in camera frame, (n, 3) array
    """
    # rays in camera frame
    n = dsize ** 2
    rays = np.zeros((n, 3))

    if pattern == 'grid':
        x = np.linspace(-fov_width/2, fov_width/2, dsize)
        y = np.linspace(-fov_width/2, fov_width/2, dsize)
        yy, xx = np.meshgrid(y, x)
        rays[:, 0] = xx.flatten()
        rays[:, 1] = yy.flatten()
        rays[:, 2] = alt
    elif pattern == "lissajous":
        r = fov_width / 4
        t = np.linspace(0, 300, n)
        w1 = 1
        w2 = -np.sqrt(2)
        x = r * np.cos(w1 * t) + r * np.cos(w2 * t)
        y = r * np.sin(w1 * t) + r * np.sin(w2 * t)
        rays[:, 0] = x
        rays[:, 1] = y
        rays[:, 2] = alt
    else:
        raise ValueError("Invalid pattern")

    return rays


def scan_dtm(dtm: Trimesh, src: np.ndarray, rays: np.ndarray, R_C2W: np.ndarray, sigma: float=0.0, seed: int=0):
    """
    Scan DTM with rays

    Args:
        dtm (trimesh.base.Trimesh): Digital Terrain Model
        rays (np.ndarray): Rays in camera frame
        R_C2W (np.ndarray): Rotation matrix from camera to world frame
        sigma (float, optional): Noise standard deviation. Defaults to 0.0.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        np.ndarray: Point cloud data
    """
    # transform rays to world frame
    rays = R_C2W @ rays.T
    rays = rays.T

    print(rays.shape)

    # normalize rays
    rays = rays / np.linalg.norm(rays, axis=1).reshape(-1, 1)

    if src.ndim == 1:
        src = np.array([src] * len(rays)).reshape(-1, 3)

    pcd, _, _ = dtm.ray.intersects_location(src, rays, multiple_hits=False)

    # add noise
    if sigma > 0:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, sigma, pcd.shape)
        pcd += noise

    return pcd