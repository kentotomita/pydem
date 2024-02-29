"""
Deterministic Hazard Detection (DHD) algorithm for safety map generation.
The algorithm is based on [ALHAT](https://arc.aiaa.org/doi/pdf/10.2514/6.2013-5019).
"""

from numba import jit, prange
import math

import numpy as np

from ._util import (discretize_lander_geom, max_under_pad, pad_pix_locations, 
                    footprint_checker, cross_product, dot_product)


def dhd(
    dem: np.ndarray,
    rmpp: float,
    negative_rghns_unsafe: bool = True,
    lander_type: str= 'square',
    dl: float = 10.0,
    dp: float = 1.0,
    scrit: float = 10 * np.pi / 180,
    rcrit: float = 0.2,
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
        site_slope (np.ndarray): slope (rad) for each landing site
        site_rghns (np.ndarray): roughness for each landing site (m)
        pix_rghns (np.ndarray): roughness for each pixel
        is_safe (np.ndarray): safety map
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
    site_slope, site_rghns, pix_rghns = dem_slope_rghns(s_rlander, s_rpad, lander_type, rmpp, negative_rghns_unsafe, xi_arr, yi_arr, dem, fpmap)

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


def dem_slope_rghns(s_rlander: int, lander_type: str, rmpp: float, 
                    negative_rghns_unsafe:bool, xi_arr: np.ndarray, yi_arr: np.ndarray,
                    dem: np.ndarray, fpmap: np.ndarray):
    """compute slope and roughness over the DEM
    Args:
        n_rlander: number of pixels to represent radius of lander
        n_rpad: number of pixels to represent radius of landing pad
        lander_type: "triangle" or "square"
        rmpp: resolution (m/pix)
        negative_rghns_unsafe: if True, negative roughness is considered as unsafe
        xi_arr: (n, 4), relative x pixel locations of landing pads
        yi_arr: (n, 4), relative y pixel locations of landing pads
        dem: (H x W), digital elevation map
        fpmap: (H x W), max values under landing pad for each pixel
    Returns:
        site_slope: (H x W), slope (rad) for each landing site
        site_rghns: (H x W), roughness for each landing site
        pix_rghns: roughness for each pixel
    """
    nr, nc = dem.shape
    nt = xi_arr.shape[0]

    site_slope = np.zeros_like(dem)
    site_slope[:] = np.nan
    site_rghns = np.zeros_like(dem)
    site_rghns[:] = np.nan
    pix_rghns = np.zeros_like(dem)
    pix_rghns[:] = -1

    footprint_mask = footprint_checker(lander_type, xi_arr, yi_arr, s_rlander)

    site_slope, site_rghns, pix_rghns = _dem_slope_rghns(nr, nc, nt, s_rlander, lander_type, rmpp, negative_rghns_unsafe,
                                                        xi_arr, yi_arr, footprint_mask, dem, fpmap,
                                                        site_slope, site_rghns, pix_rghns)

    pix_rghns[pix_rghns==-1] = np.nan
    return site_slope, site_rghns, pix_rghns



@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _dem_slope_rghns(nr:int, nc:int, nt:int, s_rlander:int, lander_type:str, rmpp:float, negative_rghns_unsafe: bool,
                     xi_arr:np.ndarray, yi_arr:np.ndarray, footprint_mask:np.ndarray, dem:np.ndarray, fpmap:np.ndarray, 
                     site_slope:np.ndarray, site_rghns:np.ndarray, pix_rghns:np.ndarray):
    """
    Args:
        nr: number of rows of dem
        nc: number of columns of dem   
        nt: number of landing orientations
        s_rlander: number of pixels to represent radius of lander
        lander_type: "triangle" or "square"
        rmpp: resolution (m/pix)
        negative_rghns_unsafe: if True, negative roughness is considered as unsafe
        xi_arr: (n, 4), relative x pixel locations of landing pads
        yi_arr: (n, 4), relative y pixel locations of landing pads
        dem: (H x W), digital elevation map
        fpmap: (H x W), max values under landing pad for each pixel
        site_slope: (H x W), slope (rad) for each landing site
        site_rghns: (H x W), roughness for each landing site
        pix_rghns: roughness for each pixel
    """

    for xi in prange(nr):
        for yi in prange(nc):

            # pixels close to edge cannot be evaluated
            if xi >= s_rlander and xi < nr - s_rlander and yi >= s_rlander and yi < nc - s_rlander:

                # Below, we use a coordinate system whose origin is located at (xi, yi, 0) in dem coordinate.
                # 1. find max slope for dem[xi, yi]
                slope = -1
                rghns = -1
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
                    
                    #a = (bottom_y - left_y) * (top_z - left_z) - (top_y - left_y) * (bottom_z - left_z)
                    #b = (bottom_z - left_z) * (top_x - left_x) - (top_z - left_z) * (bottom_x - left_x)
                    #c = (bottom_x - left_x) * (top_y - left_y) - (top_x - left_x) * (bottom_y - left_y)
                    norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
                    if norm==0:
                        print("norm==0 detected")
                        print(ti)
                        print(xi_arr[ti])
                        print(yi_arr[ti])
                        print(top_z, left_z, bottom_z, right_z)
                    a, b, c = a/norm, b/norm, c/norm

                    d = -dot_product(a, b, c, left_x, left_y, left_z)
                    #d = - a * left_x - b * left_y - c * left_z
                    # If the dot product with [a,b,c] is larger than -d, the point is above the plane.

                    # calculate gradient in rad
                    slope_th = abs(math.acos(c))
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
                                            rghns = math.nan
                                        else:
                                            if negative_rghns_unsafe:
                                                rghns_th_ij = abs(a * xp + b * yp + c * zp + d)
                                            else:
                                                rghns_th_ij = max(0, a * xp + b * yp + c * zp + d)

                                            # update cite roughness with the largest one
                                            if rghns_th_ij > rghns:
                                                rghns = rghns_th_ij
                                            
                                            # update pixel roughness with the largest one
                                            if rghns_th_ij > pix_rghns[xi + i - s_rlander, yi + j - s_rlander]:
                                                pix_rghns[xi + i - s_rlander, yi + j - s_rlander] = rghns_th_ij

                # 3. substitution
                if slope == -1:
                    slope = math.nan
                if rghns == -1:
                    rghns = math.nan
                site_slope[xi, yi] = slope
                site_rghns[xi, yi] = rghns
    return site_slope, site_rghns, pix_rghns