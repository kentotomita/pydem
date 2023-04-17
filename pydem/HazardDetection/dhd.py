"""
Deterministic Hazard Detection (DHD) algorithm for safety map generation.
The algorithm is based on [ALHAT](https://arc.aiaa.org/doi/pdf/10.2514/6.2013-5019).
"""

from numba import jit
import math

import numpy as np

from ._util import max_under_pad, dem_slope_rghns, pad_pix_locations


def dhd(
    dem: np.ndarray,
    rmpp: float,
    negative_rghns_unsafe: bool = False,
    lander_type: str= 'square',
    dl: float = 10.0,
    dp: float = 1.0,
    scrit: float = 10 * np.pi / 180,
    rcrit: float = 0.2,
    thstep: float = 10 * np.pi / 180,
    verbose: int = 0
):
    """Deterministic Hazard Detection (DHD) algorithm for safety map generation.

    Args:
        dem (np.ndarray): digital elevation map to be processed
        rmpp (float): resolution in meter per pixel
        dl (float): diameter of the lander
        dp (float): diameter of the landing pad
        scrit (float): critical slope, slope larger than this value is recognized as hazard (rad)
        rcrit (float): critical roughness, surface features larger than this value is recognized as hazard (m)

    Returns:
        fpmap (np.ndarray): footpad map
        slope (np.ndarray): maximum slope map (rad)
        rghns (np.ndarray): maximum roughness map (m)
        is_safe (np.ndarray): safety map
        indef (np.ndarray): indicates indefinite regions
    """
    assert lander_type == "triangle" or lander_type == "square"

    N_RPAD = int((dp / 2) / rmpp)  # number of pixels for radius of the landing pad; floor
    N_RLANDER = int((dl / 2) / rmpp)  # number of pixels for radius of lander; floor
    if verbose:
        print("pad radius (px):", N_RPAD)
        print("lander radius (px):", N_RLANDER)

    # 1. generate foot pad map #############################
    # prep foot pad map
    fpmap = max_under_pad(N_RPAD, dem)

    # 2. generate slope and roughness map ################################
    theta_arr = np.arange(0, np.pi * 2, thstep)
    xi_arr, yi_arr, x_arr, y_arr = pad_pix_locations(theta_arr, lander_type, rmpp, N_RLANDER, N_RPAD)
    site_slope, site_rghns, pix_rghns = dem_slope_rghns(N_RLANDER, N_RPAD, lander_type, rmpp, negative_rghns_unsafe, xi_arr, yi_arr, x_arr, y_arr, dem, fpmap)

    # make indefinite map
    indef = np.zeros_like(dem).astype(np.int)
    indef[np.isnan(site_slope)] = 1
    indef[np.isnan(site_rghns)] = 1

    # make safety map
    is_safe =  np.zeros_like(dem).astype(np.int)
    is_safe[(site_slope < scrit) * (site_rghns < rcrit)] = 1
    is_safe[indef == 1] = 0
    if verbose:
        print("MAX SLOPE:    ", np.nanmax(site_slope))
        print("MAX ROUGHNESS:", np.nanmax(site_rghns))

    return fpmap, site_slope, site_rghns, pix_rghns, is_safe, indef
