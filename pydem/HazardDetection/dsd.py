"""
Deterministic Slope Hazard Detection (DHD) algorithm for safety map generation.
"""

from numba import jit
import math

import numpy as np


def dsd(
    dem: np.ndarray,
    rmpp: float,
    dl: float,
    dp: float,
    scrit: float,
    thstep: float = 10 * np.pi / 180,
    verbose: int = 1,
):
    """Deterministic Slope Hazard Detection algorithm for safety map generation.

    Args:
        dem (np.ndarray): digital elevation map to be processed
        rmpp (float): resolution in meter per pixel
        dl (float): diameter of the lander
        dp (float): diameter of the landing pad

    Returns:
        fpmap (np.ndarray): footpad map
        slope (np.ndarray): maximum slope map (rad)
    """
    # dem = np.ascontiguousarray(dem)
    N_RPAD = np.int(dp / 2 / rmpp)  # number of pixels for radius of the landing pad
    N_RLANDER = np.int(dl / 2 / rmpp)  # number of pixels for radius of lander
    if verbose:
        print("pad radius (px):", N_RPAD)
        print("lander radius (px):", N_RLANDER)

    # 1. generate foot pad map #############################
    # prep foot pad map
    fpmap = np.ones(np.shape(dem)).astype(np.float32) * (0 - np.Inf)
    fpmap = max_under_pad(N_RPAD, dem, fpmap)

    # 2. generate slope and roughness map ################################
    # prep rcos and rsin list
    theta_list = list(np.arange(0, np.pi * 2, thstep))
    n_rcos_arr = ((N_RLANDER - N_RPAD) * np.cos(theta_list)).astype(np.int)
    n_rsin_arr = ((N_RLANDER - N_RPAD) * np.sin(theta_list)).astype(np.int)
    # prep slope and roughness map
    slope = np.ones(np.shape(dem)).astype(np.float32) * np.nan
    # run
    slope = process_dem(N_RLANDER, n_rcos_arr, n_rsin_arr, dem, rmpp, fpmap, slope)
    
    # make indefinite map
    indef = np.zeros(np.shape(dem)).astype(np.int)
    indef[np.isnan(slope)] = 1

    if verbose:
        print("MAX SLOPE:    ", np.nanmax(slope))

    return fpmap, slope, indef


@jit
def max_under_pad(n_rpad, arr_in, arr_out):
    """
    output 2d array elements have max values in nearby circular area with size of landing pad
    :param n_rpad: number of pixels to represent radius of landing pad
    :param arr_in: 2d array, (H + 2*n_rpad) x (W + 2*n_rpad)
    :param arr_out: 2d array, H x W
    :return:
    """
    for xi in range(arr_in.shape[0]):
        for yi in range(arr_in.shape[1]):

            # pixels close to edge cannot be evaluated
            if xi >= n_rpad and xi < arr_in.shape[0] - n_rpad and yi >= n_rpad and yi < arr_in.shape[1] - n_rpad:

                # find max under circular landing pad
                val = -np.Inf
                for i in range(-n_rpad, n_rpad + 1):
                    for j in range(-n_rpad, n_rpad + 1):

                        if i ** 2 + j ** 2 <= n_rpad ** 2:  # check if (i,j) is under the pad disk
                            if val < arr_in[xi + i, yi + j]:
                                val = arr_in[xi + i, yi + j]

                if math.isinf(val):
                    val = np.nan
                arr_out[xi, yi] = val
    return arr_out


@jit
def process_dem(n_lander, n_rcos_arr, n_rsin_arr, dem, resoulution_mpp, fpmap, slope_map):
    """
    :param n_lander: number of pixels to represent radius of lander
    :param n_rpad: number of pixels to represent radius of landing pad
    :param th_step: step size of lander orientation angle [rad]
    :param dem: digital elevation map
    :param fpmap: footpad map
    :param slope_map: slope map (output)
    :param rghns_map: roughness map (output)
    """

    nr, nc = dem.shape

    for xi in range(nr):
        for yi in range(nc):

            # pixels close to edge cannot be evaluated
            if xi >= n_lander and xi < nr - n_lander and yi >= n_lander and yi < nc - n_lander:

                # Below, we use a coordinate system whose origin is located at (xi, yi, 0) in dem coordinate.
                # 1. find max slope for dem[xi, yi]
                slope = -np.inf
                rghns = -np.inf
                a, b, c, d = 0, 0, 0, 0
                #'''
                for theta_i in range(n_rcos_arr.shape[0]):
                    n_rcos = n_rcos_arr[theta_i]
                    n_rsin = n_rsin_arr[theta_i]
                    rcos = n_rcos * resoulution_mpp
                    rsin = n_rsin * resoulution_mpp

                    # + ---------> y (axis 1)
                    # |
                    # |
                    # v
                    # x (axis 0)
                    # Theta: counter clockwise
                    # top pad:    [-rcos, -rsin, pad_t]
                    # left pad:   [ rsin, -rcos, pad_l]
                    # bottom pad: [ rcos,  rsin, pad_b]
                    # right pad:  [-rsin,  rcos, pad_r]

                    top_xi, top_yi = -n_rcos, -n_rsin
                    left_xi, left_yi = n_rsin, -n_rcos
                    bottom_xi, bottom_yi = n_rcos, n_rsin
                    right_xi, right_yi = -n_rsin, n_rcos

                    top_x, top_y = -rcos, -rsin
                    left_x, left_y = rsin, -rcos
                    bottom_x, bottom_y = rcos, rsin
                    right_x, right_y = -rsin, rcos

                    # get altitude of each landing pad
                    pad_t = fpmap[xi + top_xi, yi + top_yi]  # altitude of top landing pad
                    pad_l = fpmap[xi + left_xi, yi + left_yi]  # altitude of left landing pad
                    pad_b = fpmap[xi + bottom_xi, yi + bottom_yi]  # altitude of bottom landing pad
                    pad_r = fpmap[xi + right_xi, yi + right_yi]  # altitude of right landing pad

                    # set the bottom landing pad being unused for slope evaluation

                    #
                    # v1: vector right pad w.r.t. left pad
                    v1x = right_x - left_x
                    v1y = right_y - left_y
                    v1z = pad_r - pad_l
                    #
                    # v2: vector top pad w.r.t. left pad
                    v2x = top_x - left_x
                    v2y = top_y - left_y
                    v2z = pad_t - pad_l
                    #
                    # cross product to get normal vector
                    nx = v1y * v2z - v2y * v1z
                    ny = v1z * v2x - v2z * v1x
                    nz = v1x * v2y - v2x * v1y
                    n = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
                    a = nx / n
                    b = ny / n
                    c = nz / n
                    d = -right_x * a - right_y * b - pad_r * c  # using right landing pad position to calc d
                    #
                    # calculate gradient in rad
                    slope_th = abs(math.acos(c))
                    # if any of the landing pad has an altitude of NaN, set slope np.inf
                    if math.isnan(pad_t) or math.isnan(pad_l) or math.isnan(pad_b) or math.isnan(pad_r):
                        slope = np.inf
                    # update slope and roughness only if the bottom pad does not dig deep into the surface
                    elif pad_b <= -(a * bottom_x + b * bottom_y + d) / c:
                        # update slope with the largest one
                        if slope_th > slope:
                            slope = slope_th

                #'''
                # 3. substitution
                if math.isinf(slope):
                    slope = np.nan
                if math.isinf(rghns):
                    rghns = np.nan
                slope_map[xi, yi] = slope
    return slope_map

@jit
def inside_line(x, y, x0, y0, x1, y1):
    v0x = x - x0
    v0y = y - y0
    v1x = x1 - x0
    v1y = y1 - y0
    # let v2 be cross product (v1 x v0)
    v2z = v1x * v0y - v1y * v0x
    return v2z > 0  # if True, (x,y) is located to the left of the line from (x0, y0) to (x1, y1)
