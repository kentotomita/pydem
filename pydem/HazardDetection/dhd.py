"""
Deterministic Hazard Detection (DHD) algorithm for safety map generation.
The algorithm is based on [ALHAT](https://arc.aiaa.org/doi/pdf/10.2514/6.2013-5019).
"""

from numba import jit
import math

import numpy as np


def dhd(
    dem: np.ndarray,
    rmpp: float,
    lander_type: str,
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

    N_RPAD = np.int(dp / 2 / rmpp)  # number of pixels for radius of the landing pad
    N_RLANDER = np.int(dl / 2 / rmpp)  # number of pixels for radius of lander
    if verbose:
        print("pad radius (px):", N_RPAD)
        print("lander radius (px):", N_RLANDER)

    # 1. generate foot pad map #############################
    # prep foot pad map
    fpmap = np.ones(np.shape(dem)).astype(np.float64) * (0 - np.Inf)
    fpmap = max_under_pad(N_RPAD, dem, fpmap)

    # 2. generate slope and roughness map ################################
    # prep rcos and rsin list
    theta_arr = np.arange(0, np.pi * 2, thstep)
    xi_arr, yi_arr, x_arr, y_arr = pad_pix_locations(theta_arr, lander_type, rmpp, N_RLANDER, N_RPAD)
    # prep slope and roughness map
    slope = np.ones(np.shape(dem)).astype(np.float64) * np.nan
    rghns = np.ones(np.shape(dem)).astype(np.float64) * np.nan
    # run
    slope, rghns = process_dem(N_RLANDER, lander_type, rmpp, xi_arr, yi_arr, x_arr, y_arr, dem, fpmap, slope, rghns)

    # make indefinite map
    indef = np.zeros(np.shape(dem)).astype(np.int)
    indef[np.isnan(slope)] = 1
    indef[np.isnan(rghns)] = 1

    # make safety map
    is_safe = np.zeros(np.shape(dem)).astype(np.int)
    is_safe[(slope < scrit) * (rghns < rcrit)] = 1
    is_safe[indef == 1] = 0
    if verbose:
        print("MAX SLOPE:    ", np.nanmax(slope))
        print("MAX ROUGHNESS:", np.nanmax(rghns))

    return fpmap, slope, rghns, is_safe, indef


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
def process_dem(n_lander, lander_type, rmpp, xi_arr, yi_arr, x_arr, y_arr, dem, fpmap, slope_map, rghns_map):
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
    n_theta = xi_arr.shape[0]

    for xi in range(nr):
        for yi in range(nc):

            # pixels close to edge cannot be evaluated
            if xi >= n_lander and xi < nr - n_lander and yi >= n_lander and yi < nc - n_lander:

                # Below, we use a coordinate system whose origin is located at (xi, yi, 0) in dem coordinate.
                # 1. find max slope for dem[xi, yi]
                slope = -np.inf
                rghns = -np.inf
                #'''
                for theta_i in range(n_theta):

                    left_xi, bottom_xi, top_xi, right_xi = xi_arr[theta_i]
                    left_yi, bottom_yi, top_yi, right_yi = yi_arr[theta_i]
                    left_x, bottom_x, top_x, right_x = x_arr[theta_i]
                    left_y, bottom_y, top_y, right_y = y_arr[theta_i]

                    # get altitude of each landing pad
                    top_z = fpmap[xi + top_xi, yi + top_yi]  # altitude of top landing pad
                    left_z = fpmap[xi + left_xi, yi + left_yi]  # altitude of left landing pad
                    bottom_z = fpmap[xi + bottom_xi, yi + bottom_yi]  # altitude of bottom landing pad
                    right_z = fpmap[xi + right_xi, yi + right_yi]  # altitude of right landing pad

                    # -------------------------------------------
                    # Use (top, left, bottom) landing pads for computation
                    # -------------------------------------------
                    a = (bottom_y - left_y) * (top_z - left_z) - (top_y - left_y) * (bottom_z - left_z)
                    b = (bottom_z - left_z) * (top_x - left_x) - (top_z - left_z) * (bottom_x - left_x)
                    c = (bottom_x - left_x) * (top_y - left_y) - (top_x - left_x) * (bottom_y - left_y)

                    norm = np.sqrt(a ** 2 + b ** 2 + c ** 2)
                    if norm==0:
                        print(theta_i)
                        print(xi_arr[theta_i])
                        print(yi_arr[theta_i])
                        print(top_z, left_z, bottom_z, right_z)
                    a, b, c = a/norm, b/norm, c/norm
                    d = - a * left_x - b * left_y - c * left_z

                    # calculate gradient in rad
                    slope_th = abs(math.acos(c))
                    # if any of the landing pad has an altitude of NaN, set slope np.inf
                    if math.isnan(top_z) or math.isnan(left_z) or math.isnan(bottom_z) or math.isnan(right_z):
                        slope = np.inf

                    cond1 = lander_type == "triangle"
                    cond2 = lander_type == "square"
                    cond3 = right_z <= -(a * right_x + b * right_y + d) / c  # right pad does not dig deep into the surface
                    if cond1 or (cond2 and cond3):
                        # update slope with the largest one
                        if slope_th > slope:
                            slope = slope_th

                        # 2. find maximum roughness for dem[i, j] for all theta
                        for i in range(-n_lander, n_lander + 1):
                            for j in range(-n_lander, n_lander + 1):
                                xp = i * rmpp
                                yp = j * rmpp

                                cond1 = inside_line(xp, yp, top_x, top_y, left_x, left_y)
                                cond2 = inside_line(xp, yp, left_x, left_y, bottom_x, bottom_y)
                                cond3 = inside_line(xp, yp, bottom_x, bottom_y, right_x, right_y)
                                cond4 = inside_line(xp, yp, right_x, right_y, top_x, top_y)
                                cond5 = inside_line(xp, yp, bottom_x, bottom_y, top_x, top_y)

                                if lander_type=="triangle":
                                    within_footprint = (cond1 and cond2 and cond5)
                                else:
                                    within_footprint = (cond1 and cond2 and cond3 and cond4)

                                #if i ** 2 + j ** 2 <= n_lander ** 2:  # check if under the circular lander footprint
                                if within_footprint:
                                    zp = dem[xi + i, yi + j]

                                    # if NaN, set roughness infinite
                                    if math.isnan(zp):
                                        rghns = np.inf
                                    else:
                                        rghns_th_ij = abs(a * xp + b * yp + c * zp + d)
                                        if rghns_th_ij > rghns:
                                            rghns = rghns_th_ij
                #'''
                # 3. substitution
                if math.isinf(slope):
                    slope = np.nan
                if math.isinf(rghns):
                    rghns = np.nan
                slope_map[xi, yi] = slope
                rghns_map[xi, yi] = rghns
    return slope_map, rghns_map


@jit
def inside_line(x, y, x0, y0, x1, y1):
    """Return True if (x, y) is on the left of the line connecting (x0, y0) and (x1, y1)"""
    v0x = x - x0
    v0y = y - y0
    v1x = x1 - x0
    v1y = y1 - y0
    # let v2 be cross product (v1 x v0)
    v2z = v1x * v0y - v1y * v0x
    return v2z > 0  # if True, (x,y) is located to the left of the line from (x0, y0) to (x1, y1)


@jit
def pad_pix_locations(theta_arr, lander_type, rmpp: float, N_RLANDER, N_RPAD):
    """Return the list of relative locations (pix) of landing pads"""

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
    x_arr = np.zeros(shape=(n, 4)).astype(np.float32)
    y_arr = np.zeros(shape=(n, 4)).astype(np.float32)

    R = N_RLANDER - N_RPAD
    for i in range(n):
        th = theta_arr[i]
        n_rcos = int(R * np.cos(th))
        n_rsin = int(R * np.sin(th))
        if lander_type=="triangle":
            # pad 0: left pad
            # pad 1: bottom right pad
            # pad 2: top right pad
            xi_0, yi_0 = int(R * np.sin(th)), int(R * np.cos(th + np.pi))
            xi_1, yi_1 = int(R * np.sin(th + np.pi/3)), int(R * np.cos(th + np.pi * 4/3))
            xi_2, yi_2 = int(R * np.sin(th - np.pi/3)), int(R * np.cos(th + np.pi * 2/3))
            xi_3, yi_3 = 0, 0  # dummy
        else:  # lander_type=="square":
            # pad 0: left pad
            # pad 1: bottom pad
            # pad 2: top pad
            # pad 3: right pad

            #xi_0, yi_0 = int(R * np.sin(th)), int(R * np.cos(th + np.pi))
            #xi_1, yi_1 = int(R * np.sin(th + np.pi / 2)), int(R * np.cos(th + np.pi * 3 / 2))
            #xi_2, yi_2 = int(R * np.sin(th - np.pi / 2)), int(R * np.cos(th + np.pi * 1 / 2))
            #xi_3, yi_3 = int(R * np.sin(th + np.pi)), int(R * np.cos(th))
            xi_0, yi_0 =  n_rsin, -n_rcos
            xi_1, yi_1 =  n_rcos,  n_rsin
            xi_2, yi_2 = -n_rcos, -n_rsin
            xi_3, yi_3 = -n_rsin,  n_rcos



        xi_arr[i] = np.array([xi_0, xi_1, xi_2, xi_3]).astype(np.int32)
        yi_arr[i] = np.array([yi_0, yi_1, yi_2, yi_3]).astype(np.int32)
        x_arr[i] = (xi_arr[i] * rmpp).astype(np.float32)
        y_arr[i] = (yi_arr[i] * rmpp).astype(np.float32)

    return xi_arr, yi_arr, x_arr, y_arr
