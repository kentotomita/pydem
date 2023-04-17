"""Utility functions for Hazard Detection module."""

import numpy as np
import math
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


def dem_slope_rghns(n_rlander: int, n_rpad: int, lander_type: str, rmpp: float, negative_rghns_unsafe:bool, 
                    xi_arr: np.ndarray, yi_arr: np.ndarray, x_arr: np.ndarray, y_arr: np.ndarray, dem: np.ndarray, fpmap: np.ndarray):
    """compute slope and roughness over the DEM
    Args:
        n_rlander: number of pixels to represent radius of lander
        n_rpad: number of pixels to represent radius of landing pad
        lander_type: "triangle" or "square"
        rmpp: resolution (m/pix)
        negative_rghns_unsafe: if True, negative roughness is considered as unsafe
        xi_arr: (n, 4), relative x pixel locations of landing pads
        yi_arr: (n, 4), relative y pixel locations of landing pads
        x_arr: (n, 4), relative x locations of landing pads
        y_arr: (n, 4), relative y locations of landing pads
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

    site_slope, site_rghns, pix_rghns = _dem_slope_rghns(nr, nc, nt, n_rlander, n_rpad, lander_type, rmpp, negative_rghns_unsafe,
                                                        xi_arr, yi_arr, x_arr, y_arr, dem, fpmap,
                                                        site_slope, site_rghns, pix_rghns)

    pix_rghns[pix_rghns==-1] = np.nan
    return site_slope, site_rghns, pix_rghns



@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _dem_slope_rghns(nr:int, nc:int, nt:int, n_rlander:int, n_rpad: int, lander_type:str, rmpp:float, negative_rghns_unsafe: bool,
                     xi_arr:np.ndarray, yi_arr:np.ndarray, x_arr:np.ndarray, y_arr:np.ndarray, dem:np.ndarray, fpmap:np.ndarray, 
                     site_slope:np.ndarray, site_rghns:np.ndarray, pix_rghns:np.ndarray):
    """
    Args:
        nr: number of rows of dem
        nc: number of columns of dem   
        nt: number of landing orientations
        n_rlander: number of pixels to represent radius of lander
        n_rpad: number of pixels to represent radius of landing pad
        lander_type: "triangle" or "square"
        rmpp: resolution (m/pix)
        negative_rghns_unsafe: if True, negative roughness is considered as unsafe
        xi_arr: (n, 4), relative x pixel locations of landing pads
        yi_arr: (n, 4), relative y pixel locations of landing pads
        x_arr: (n, 4), relative x locations of landing pads
        y_arr: (n, 4), relative y locations of landing pads
        dem: (H x W), digital elevation map
        fpmap: (H x W), max values under landing pad for each pixel
        site_slope: (H x W), slope (rad) for each landing site
        site_rghns: (H x W), roughness for each landing site
        pix_rghns: roughness for each pixel
    """

    for xi in prange(nr):
        for yi in prange(nc):

            # pixels close to edge cannot be evaluated
            if xi >= n_rlander and xi < nr - n_rlander and yi >= n_rlander and yi < nc - n_rlander:

                # Below, we use a coordinate system whose origin is located at (xi, yi, 0) in dem coordinate.
                # 1. find max slope for dem[xi, yi]
                slope = -1
                rghns = -1
                # loop over all landing orientations; angle theta (rad)
                for ti in prange(nt):

                    left_xi, bottom_xi, right_xi, top_xi = xi_arr[ti]
                    left_yi, bottom_yi, right_yi, top_yi = yi_arr[ti]
                    left_x, bottom_x, right_x, top_x, = x_arr[ti]
                    left_y, bottom_y, right_y, top_y = y_arr[ti]

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
                            for i in prange(-n_rlander + (n_rpad*2 + 1), n_rlander - (n_rpad*2 + 1) + 1):
                                for j in prange(-n_rlander + (n_rpad*2 + 1), n_rlander - (n_rpad*2 + 1) + 1):
                                    xp = i * rmpp
                                    yp = j * rmpp

                                    cond1 = inside_line(xp, yp, top_x, top_y, left_x, left_y)
                                    cond2 = inside_line(xp, yp, left_x, left_y, bottom_x, bottom_y)
                                    cond3 = inside_line(xp, yp, bottom_x, bottom_y, right_x, right_y)
                                    cond4 = inside_line(xp, yp, right_x, right_y, top_x, top_y)
                                    cond5 = inside_line(xp, yp, right_x, right_y, left_x, left_y)

                                    if lander_type=="triangle":
                                        within_footprint = (cond2 and cond3 and cond5)
                                    else:
                                        within_footprint = (cond1 and cond2 and cond3 and cond4)

                                    if within_footprint:
                                        zp = dem[xi + i, yi + j]

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
                                            if rghns_th_ij > pix_rghns[xi + i, yi + j]:
                                                pix_rghns[xi + i, yi + j] = rghns_th_ij
                # 3. substitution
                if slope == -1:
                    slope = math.nan
                if rghns == -1:
                    rghns = math.nan
                site_slope[xi, yi] = slope
                site_rghns[xi, yi] = rghns
    return site_slope, site_rghns, pix_rghns


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
        # pad 3: dummy
        #       (2)
        #     /  |
        # (0)    |
        #     \  |
        #       (1)
        xi_0, yi_0 = int(R * np.sin(th)), int(R * np.cos(th + np.pi))
        xi_1, yi_1 = int(R * np.sin(th + np.pi * 2/3)), int(R * np.cos(th - np.pi/3))
        xi_2, yi_2 = int(R * np.sin(th - np.pi * 2/3)), int(R * np.cos(th + np.pi/3))
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

        # pad 0: left pad;   (sin(th),        cos(th + pi))   = (sin(th), -cos(th)) 
        # pad 1: bottom pad; (sin(th + pi/2), cos(th - pi/2)) = (cos(th),  sin(th))
        # pad 2: right pad;  (sin(th + pi),   cos(th))        = (-sin(th), cos(th))
        # pad 3: top pad;    (sin(th - pi/2), cos(th + pi/2)) = (-cos(th), -sin(th))
        #     (3)
        #   /     \
        # (0)     (2)
        #   \     /
        #     (1)
        xi_0, yi_0 =  n_rsin, -n_rcos
        xi_1, yi_1 =  n_rcos,  n_rsin
        xi_2, yi_2 = -n_rsin,  n_rcos
        xi_3, yi_3 = -n_rcos, -n_rsin

        xi_arr[i] = np.array([xi_0, xi_1, xi_2, xi_3]).astype(np.int32)
        yi_arr[i] = np.array([yi_0, yi_1, yi_2, yi_3]).astype(np.int32)

    return xi_arr, yi_arr


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def inside_line(x, y, x0, y0, x1, y1):
    """Return True if (x, y) is on the left of the line connecting from (x0, y0) to (x1, y1)"""
    v0x = x - x0
    v0y = y - y0
    v1x = x1 - x0
    v1y = y1 - y0
    # let v2 be cross product (v1 x v0)
    v2z = v1x * v0y - v1y * v0x
    return v2z > 0  # if True, (x,y) is located to the left of the line from (x0, y0) to (x1, y1)



@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def cross_product(v1x, v1y, v1z, v2x, v2y, v2z):
    """Return cross product of v1 and v2"""
    v3x = v1y * v2z - v1z * v2y
    v3y = v1z * v2x - v1x * v2z
    v3z = v1x * v2y - v1y * v2x 
    return v3x, v3y, v3z


@jit(nopython=True, fastmath=True, nogil=True, cache=True)
def dot_product(v1x, v1y, v1z, v2x, v2y, v2z):
    """Return dot product of v1 and v2"""
    return v1x * v2x + v1y * v2y + v1z * v2z


