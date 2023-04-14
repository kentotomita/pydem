"""Utility functions for Hazard Detection module."""

import numpy as np
from numba import jit, prange


def max_under_pad(n_rpad:int, dem:np.ndarray) -> np.ndarray:
    """Return max values under landing pad for each pixel. 
    Args:
        n_rpad: number of pixels to represent radius of landing pad
        dem: (H x W), digital elevation map
    Returns:
        fpmap: (H x W), max values under landing pad for each pixel
    """
    zmin = np.nanmin(dem)
    dem = dem - zmin

    fpmap = np.zeros_like(dem)
    fpmap[:] = np.nan
    fpmap = _max_under_pad(dem.shape[0], dem.shape[1], n_rpad, dem, fpmap)

    return fpmap + zmin


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _max_under_pad(h:int, w:int, n_rpad:int, dem:np.ndarray, fpmap:np.ndarray):
    """Return max values under landing pad for each pixel. dem has to be >0.
    Args:
        n_rpad: number of pixels to represent radius of landing pad
        dem: >0, (H x W), digital elevation map
        fpmap: (H x W), max values under landing pad for each pixel
    """
    for xi in prange(h):
        for yi in prange(w):

            # pixels close to edge cannot be evaluated
            if xi >= n_rpad and xi < h - n_rpad and yi >= n_rpad and yi < w - n_rpad:

                # find max under circular landing pad
                val = -1
                for i in prange(-n_rpad, n_rpad + 1):
                    for j in prange(-n_rpad, n_rpad + 1):

                        if i ** 2 + j ** 2 <= n_rpad ** 2:  # check if (i,j) is under the pad disk
                            if val < dem[xi + i, yi + j]:
                                val = dem[xi + i, yi + j]
                
                fpmap[xi, yi] = val
    return fpmap



def pad_pix_locations(theta_arr:np.ndarray, lander_type:str, rmpp: float, N_RLANDER:int, N_RPAD:int):
    """Return the list of relative locations (pix) of landing pads
    Args:
        theta_arr: (n,), angle of landing orientation
        lander_type: "triangle" or "square"
        rmpp: resolution (m/pix)
        N_RLANDER: number of pixels to represent radius of lander
        N_RPAD: number of pixels to represent radius of landing pad
    """
    # + ---------> y (axis 1)
    # |
    # |
    # v
    # x (axis 0)
    # Theta: counter clockwise

    assert lander_type == "triangle" or lander_type == "square"

    n = theta_arr.size

    xi_arr = np.zeros(shape=(n, 4)).astype(np.int32)
    yi_arr = np.zeros(shape=(n, 4)).astype(np.int32)

    if lander_type=="triangle":
        xi_arr, yi_arr, = _pad_locs_tri(n, theta_arr, xi_arr, yi_arr, N_RLANDER, N_RPAD)
    else:  # lander_type=="square":
        xi_arr, yi_arr, = _pad_locs_sq(n, theta_arr, xi_arr, yi_arr, N_RLANDER, N_RPAD)

    x_arr = (xi_arr * rmpp).astype(np.float32)
    y_arr = (yi_arr * rmpp).astype(np.float32)

    return xi_arr, yi_arr, x_arr, y_arr


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _pad_locs_tri(n:int, th_arr:np.ndarray, xi_arr:np.ndarray, yi_arr:np.ndarray, N_RLANDER:int, N_RPAD:int):
    """Return the list of relative locations (pix) of landing pads; triangle lander"""
    # + ---------> y (axis 1)
    # |
    # |
    # v
    # x (axis 0)
    # Theta: counter clockwise

    R = N_RLANDER - N_RPAD
    for i in prange(n):
        th = th_arr[i]
        n_rcos = int(R * np.cos(th))
        n_rsin = int(R * np.sin(th))

        # pad 0: left pad
        # pad 1: bottom right pad
        # pad 2: top right pad
        xi_0, yi_0 = int(R * np.sin(th)), int(R * np.cos(th + np.pi))
        xi_1, yi_1 = int(R * np.sin(th + np.pi/3)), int(R * np.cos(th + np.pi * 4/3))
        xi_2, yi_2 = int(R * np.sin(th - np.pi/3)), int(R * np.cos(th + np.pi * 2/3))
        xi_3, yi_3 = 0, 0  # dummy

        xi_arr[i] = np.array([xi_0, xi_1, xi_2, xi_3]).astype(np.int32)
        yi_arr[i] = np.array([yi_0, yi_1, yi_2, yi_3]).astype(np.int32)

    return xi_arr, yi_arr


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _pad_locs_sq(n:int, th_arr:np.ndarray, xi_arr:np.ndarray, yi_arr:np.ndarray, N_RLANDER:int, N_RPAD:int):
    """Return the list of relative locations (pix) of landing pads; square lander"""
    # + ---------> y (axis 1)
    # |
    # |
    # v
    # x (axis 0)
    # Theta: counter clockwise

    R = N_RLANDER - N_RPAD
    for i in prange(n):
        th = th_arr[i]
        n_rcos = int(R * np.cos(th))
        n_rsin = int(R * np.sin(th))

        # pad 0: left pad
        # pad 1: bottom pad
        # pad 2: top pad
        # pad 3: right pad
        xi_0, yi_0 =  n_rsin, -n_rcos
        xi_1, yi_1 =  n_rcos,  n_rsin
        xi_2, yi_2 = -n_rcos, -n_rsin
        xi_3, yi_3 = -n_rsin,  n_rcos

        xi_arr[i] = np.array([xi_0, xi_1, xi_2, xi_3]).astype(np.int32)
        yi_arr[i] = np.array([yi_0, yi_1, yi_2, yi_3]).astype(np.int32)

    return xi_arr, yi_arr