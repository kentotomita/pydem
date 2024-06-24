"""
Stochastic Hazard Detection (DHD) algorithm for safety map generation.
The algorithm is based on [ALHAT](https://arc.aiaa.org/doi/pdf/10.2514/6.2013-5019).
"""

from numba import jit, prange, float64, njit
import math
from math import erf, sqrt
import numpy as np

from ._util import (discretize_lander_geom, max_under_pad, pad_pix_locations, 
                    footprint_checker, cross_product, dot_product)


def alhat(
    dem: np.ndarray,
    rmpp: float,
    lander_type: str= 'square',
    dl: float = 10.0,
    dp: float = 1.0,
    scrit: float = 10 * np.pi / 180,
    rcrit: float = 0.2,
    sigma: float = 0.05/3,  # 5cm in 3-sigma; standard deviation of DEM noise
    verbose: int = 0
):
    """Probabilistic Hazard Detection (PHD) algorithm for safety map generation.
    The algorithm is based on [ALHAT](https://arc.aiaa.org/doi/pdf/10.2514/6.2013-5019).

    Args:
        dem (np.ndarray): noised digital elevation map to be processed
        rmpp (float): resolution in meter per pixel
        dl (float): diameter of the lander
        dp (float): diameter of the landing pad
        scrit (float): critical slope, slope larger than this value is recognized as hazard (rad)
        rcrit (float): critical roughness, surface features larger than this value is recognized as hazard (m)
        sigma (float): standard deviation of DEM noise

    Returns:
        fpmap (np.ndarray): footpad map
        site_slope (np.ndarray): slope (rad) for each landing site
        p_rghns_safe (np.ndarray): probabilistic roughness safe for each landing site (m)
        p_safe (np.ndarray): probabilistic safety map
        indef (np.ndarray): indicates indefinite regions
    """
    assert lander_type == "triangle" or lander_type == "square"

    s_rlander, s_rpad, s_radius2pad = discretize_lander_geom(dl, dp, rmpp)
    if verbose:
        print("lander radius (pix):", s_rlander)
        print("pad radius (pix):", s_rpad)

    # 1. generate foot pad map #############################
    # prep foot pad map
    fpmap = max_under_pad(dp, s_rpad, dem, rmpp)

    # 2. generate slope and roughness map ################################
    xi_arr, yi_arr = pad_pix_locations(lander_type, s_radius2pad)
    site_slope, site_prsafe, pix_prsafe = psafe_alhat(s_rlander, lander_type, rmpp,
                                                      sigma, rcrit, xi_arr, yi_arr,
                                                      dem, fpmap)
    
    # make indefinite map
    indef = np.zeros_like(dem).astype(np.int)
    indef[np.isnan(site_slope)] = 1
    indef[np.isnan(site_prsafe)] = 1

    # make safety map
    psafe =  np.zeros_like(dem).astype(np.int)
    psafe[:] = 1
    psafe[(site_slope > scrit)] = 0
    psafe = psafe * site_prsafe
    psafe[indef == 1] = np.nan
    if verbose:
        print("MAX SLOPE:    ", np.nanmax(site_slope))
        print("MIN P(ROUGHNESS SAFE):", np.nanmax(site_prsafe))

    return fpmap, site_slope, site_prsafe, pix_prsafe, psafe, indef


def psafe_alhat(s_rlander: int, lander_type: str, rmpp: float,
                sigma:float, rcrit:float, xi_arr: np.ndarray, yi_arr: np.ndarray, 
                dem: np.ndarray, fpmap: np.ndarray):
    """compute slope and roughness over the DEM
    """
    nr, nc = dem.shape
    nt = xi_arr.shape[0]

    site_slope = np.zeros_like(dem)
    site_slope[:] = np.nan
    site_prsafe = np.zeros_like(dem)
    site_prsafe[:] = np.nan
    pix_prsafe = np.zeros_like(dem)
    pix_prsafe[:] = 2

    footprint_mask = footprint_checker(lander_type, xi_arr, yi_arr, s_rlander)

    site_slope, site_prsafe, pix_prsafe = _psafe_alhat(nr, nc, nt, s_rlander, lander_type, 
                                                       rmpp, sigma, rcrit,
                                                       xi_arr, yi_arr, footprint_mask, dem, fpmap,
                                                       site_slope, site_prsafe, pix_prsafe)
    
    pix_prsafe[pix_prsafe==2] = np.nan

    return site_slope, site_prsafe, pix_prsafe


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _psafe_alhat(nr:int, nc:int, nt:int, s_rlander:int, lander_type:str, 
                 rmpp:float, sigma: float, rcrit:float, xi_arr:np.ndarray,
                 yi_arr:np.ndarray, footprint_mask: np.ndarray, dem:np.ndarray, fpmap:np.ndarray, 
                 site_slope:np.ndarray, site_prsafe:np.ndarray, pix_prsafe:np.ndarray):
    """
    Args:
        nr: number of rows of dem
        nc: number of columns of dem   
        nt: number of landing orientations
        s_rlander: number of pixels to represent radius of lander
        s_rpad: number of pixels to represent radius of landing pad
        lander_type: "triangle" or "square"
        rmpp: resolution (m/pix)
        negative_rghns_unsafe: if True, negative roughness is considered as unsafe
        xi_arr: (n, 4), relative x pixel locations of landing pads
        yi_arr: (n, 4), relative y pixel locations of landing pads
        dem: (H x W), digital elevation map
        fpmap: (H x W), max values under landing pad for each pixel
        site_slope: (H x W), slope (rad) for each landing site
        site_prsafe: (H x W), probabilistic roughness safe for each landing site (m)
        pix_prsafe: (H x W), probabilistic roughness safe for each pixel (m)
    """

    for xi in prange(nr):
        for yi in prange(nc):

            # pixels close to edge cannot be evaluated
            if xi >= s_rlander and xi < nr - s_rlander and yi >= s_rlander and yi < nc - s_rlander:

                # Below, we use a coordinate system whose origin is located at (xi, yi, 0) in dem coordinate.
                # 1. find max slope for dem[xi, yi]
                slope = -1
                prsafe = 1
                # loop over all landing orientations; angle theta (rad)
                for ti in prange(nt):

                    left_xi, bottom_xi, right_xi, top_xi = xi_arr[ti]
                    left_yi, bottom_yi, right_yi, top_yi = yi_arr[ti]
                    left_x, bottom_x, right_x, top_x, = left_xi * rmpp, bottom_xi * rmpp, right_xi * rmpp, top_xi * rmpp
                    left_y, bottom_y, right_y, top_y = left_yi * rmpp, bottom_yi * rmpp, right_yi * rmpp, top_yi * rmpp

                    # get height of each landing pad
                    left_z = fpmap[xi + left_xi, yi + left_yi]  # height of left landing pad
                    bottom_z = fpmap[xi + bottom_xi, yi + bottom_yi]  # height of bottom landing pad
                    right_z = fpmap[xi + right_xi, yi + right_yi]  # height of right landing pad
                    top_z = fpmap[xi + top_xi, yi + top_yi]  # height of top landing pad

                    # -------------------------------------------
                    # Use (left, bottom, right) landing pads for computation
                    # -------------------------------------------
                    a, b, c = cross_product(
                        v1x=bottom_x - left_x, v1y=bottom_y - left_y, v1z=bottom_z - left_z,
                        v2x=right_x - left_x, v2y=right_y - left_y, v2z=right_z - left_z
                    )
                    norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
                    if norm==0:
                        print("norm==0 detected")
                        print(ti)
                        print(xi_arr[ti])
                        print(yi_arr[ti])
                        print(top_z, left_z, bottom_z, right_z)
                    a, b, c = a/norm, b/norm, c/norm

                    d = -dot_product(a, b, c, left_x, left_y, left_z)
                    # If the dot product with [a,b,c] is larger than -d, the point is above the plane.

                    slope_th = abs(math.acos(c))  # slope (rad) for theta
                    prsafe = 1  # initialize probability of safety for theta

                    # if any of the landing pad has an height of NaN, set slope to NaN
                    if math.isnan(top_z) or math.isnan(left_z) or math.isnan(bottom_z) or math.isnan(right_z):
                        slope_th = math.nan
                    else:
                        cond1 = lander_type == "triangle"
                        cond2 = lander_type == "square"
                        # terrain height under top pad (=top_z) is lower than the landing plane (a, b, c)
                        cond3 = dot_product(a, b, c, top_x, top_y, top_z) <= -d
                        if cond1 or (cond2 and cond3):
                            # update slope with the largest one
                            if slope_th > slope:
                                slope = slope_th

                            # 2. find maximum roughness for dem[i, j] for all theta
                            for i in prange(2 * s_rlander + 1):
                                for j in prange(2 * s_rlander + 1):
                                    if footprint_mask[ti, i, j]:
                                        zp = dem[xi + i - s_rlander, yi + j - s_rlander]
                                        xp = (i - s_rlander) * rmpp
                                        yp = (j - s_rlander) * rmpp

                                        # if NaN, set roughness to NaN
                                        if math.isnan(zp):
                                            prsafe_th_ij = math.nan
                                        else:
                                            rghns_th_ij = a * xp + b * yp + c * zp + d
                                            prsafe_th_ij = cdf(rcrit, rghns_th_ij, c*sigma)
                                        
                                        #prsafe_th = prsafe_th * prsafe_th_ij  # Eq. (13) in ALHAT paper

                                        # update prsafe with the smallest one
                                        if prsafe_th_ij < prsafe:
                                            prsafe = prsafe_th_ij

                                        # update pixel roughness safety with the smallest one
                                        
                                        if prsafe_th_ij < pix_prsafe[xi + i - s_rlander, yi + j - s_rlander]:
                                            pix_prsafe[xi + i - s_rlander, yi + j - s_rlander] = prsafe_th_ij
                
                # 3. substitution
                if slope == -1:
                    slope = math.nan
                site_slope[xi, yi] = slope
                site_prsafe[xi, yi] = prsafe
    return site_slope, site_prsafe, pix_prsafe


@njit(float64(float64, float64, float64), fastmath=True)
def cdf(x, mu, sigma):
    """
    Approximated analytical function for the CDF of a normal distribution.
    
    Parameters
    ----------
    x : float
        The value at which to evaluate the CDF.
    mu : float
        The mean of the normal distribution.
    sigma : float
        The standard deviation of the normal distribution.
    
    Returns
    -------
    float
        The value of the CDF at the given value.
    """
    z = (x - mu) / (sigma * sqrt(2))
    return 0.5 * (1 + erf(z))