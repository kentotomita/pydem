"""Utility functions for Hazard Detection module."""

import numpy as np
from numba import njit, jit, prange

from . import INT, FLOAT

@jit
def discretize_lander_geom(dl: float, dp: float, rmpp: float):
    """Discretize lander geometry.
    Args:
        dl: diameter of the lander (m)
        dp: diameter of the landing pad (m)
        rmpp: resolution (m/pix)
    Returns:
        s_rlander: number of pixels to represent radius of lander
        s_rpad: number of pixels to represent radius of the landing pad
        s_radius2pad: number of pixels from the center of the lander to the center of the landing pad
    """
    s_rlander = int(np.ceil((dl/2 - rmpp/2) / rmpp))
    s_rpad = int(np.ceil(dp/rmpp) // 2)
    s_radius2pad = s_rlander - s_rpad
    assert s_radius2pad > 0, "Invalid lander size"

    return s_rlander, s_rpad, s_radius2pad


def max_under_pad(dp:float, s_rpad: int, dem: np.ndarray, rmpp: float) -> np.ndarray:
    """Return max values under landing pad for each pixel. 
    Args:
        dp: diameter of the landing pad (m)
        dem: (H x W), digital elevation map
        rmpp: resolution (m/pix)
    Returns:
        fpmap: (H x W), max values under landing pad for each pixel
    """
    zmin = np.nanmin(dem)
    dem = dem - zmin

    fpmap = np.zeros_like(dem)
    fpmap[:] = np.nan

    pad_mask, s_rpad_ = pixels_overlap_disk(dp, rmpp)
    assert s_rpad_ == s_rpad, "Landing pad diameter inconsistent"

    if s_rpad==0:
        print("Warning: landing pad diameter is too small to be represented in the DEM resolution. DEM is used as the footprint map.")
        return dem + zmin
    
    fpmap = _max_under_pad(dem.shape[0], dem.shape[1], s_rpad, pad_mask, dem, fpmap)

    return fpmap + zmin


@njit(fastmath=True, nogil=True, cache=True, parallel=True)
def _max_under_pad(h:int, w:int, s_rpad:int, pad_mask: np.ndarray, dem:np.ndarray, fpmap:np.ndarray):
    """Return max values under landing pad for each pixel. dem has to be >0.
    Args:
        n_rpad: number of pixels to represent radius of landing pad
        dem: >0, (H x W), digital elevation map
        fpmap: (H x W), max values under landing pad for each pixel
    """
    for xi in prange(h):
        for yi in prange(w):

            # pixels close to edge cannot be evaluated
            if xi >= s_rpad and xi < h - s_rpad and yi >= s_rpad and yi < w - s_rpad:

                # find max under circular landing pad
                val = -np.inf
                for i in prange(2 * s_rpad + 1):
                    for j in prange(2 * s_rpad + 1):
                        if pad_mask[i, j] == 1:
                            tmp = dem[xi + i - s_rpad, yi + j - s_rpad]
                            if val < tmp:
                                val = tmp
                
                fpmap[xi, yi] = val
    return fpmap


def pad_pix_locations(lander_type:str, s_radius2pad: float, dl: float, dp: float):
    """Return the list of relative locations (pix) of landing pads
    Args:
        lander_type: "triangle" or "square"
        s_radius2pad: pixel distance from the center of the lander to the center of the landing pad
    """
    # + ---------> y (axis 1)
    # |
    # |
    # v
    # x (axis 0)

    assert lander_type == "triangle" or lander_type == "square"
    R = (dl - dp) / 2

    circle_points = midpoint_circle(s_radius2pad)
    x_base = circle_points[:, 0]
    y_base = circle_points[:, 1]

    n = len(circle_points)
    xi_arr = np.zeros(shape=(n, 4)).astype(INT)
    yi_arr = np.zeros(shape=(n, 4)).astype(INT)
    xy_arr = np.zeros(shape=(4, 2)).astype(FLOAT)

    if lander_type=="triangle":
        # pad 0: left pad
        # pad 1: bottom right pad
        # pad 2: top right pad
        # pad 3: dummy
        #       (2)
        #     /  |
        # (0)    |
        #     \  |
        #       (1)
        step = n // 3
        xi_arr[:, 0] = x_base
        xi_arr[:, 1] = np.roll(x_base, -step)
        xi_arr[:, 2] = np.roll(x_base, -2*step)
        yi_arr[:, 0] = y_base
        yi_arr[:, 1] = np.roll(y_base, -step)
        yi_arr[:, 2] = np.roll(y_base, -2*step)

        xy_arr[0] = 0.0, -R
        xy_arr[1] = R * np.cos(np.pi / 3), R * np.sin(np.pi / 3)
        xy_arr[2] = R * np.cos(2 * np.pi / 3), -R * np.sin(2 *np.pi / 3)

    elif lander_type=="square":
        # pad 0: left pad;   (sin(th),        cos(th + pi))   = (sin(th), -cos(th)) 
        # pad 1: bottom pad; (sin(th + pi/2), cos(th - pi/2)) = (cos(th),  sin(th))
        # pad 2: right pad;  (sin(th + pi),   cos(th))        = (-sin(th), cos(th))
        # pad 3: top pad;    (sin(th - pi/2), cos(th + pi/2)) = (-cos(th), -sin(th))
        #     (3)
        #   /     \
        # (0)     (2)
        #   \     /
        #     (1)
        step = n // 4
        xi_arr[:, 0] = x_base
        xi_arr[:, 1] = np.roll(x_base, -step)
        xi_arr[:, 2] = np.roll(x_base, -2*step)
        xi_arr[:, 3] = np.roll(x_base, -3*step)
        yi_arr[:, 0] = y_base
        yi_arr[:, 1] = np.roll(y_base, -step)
        yi_arr[:, 2] = np.roll(y_base, -2*step)
        yi_arr[:, 3] = np.roll(y_base, -3*step)

        xy_arr[0] = 0.0, -R
        xy_arr[1] = R, 0.0
        xy_arr[2] = 0.0, R
        xy_arr[3] = -R, 0.0

    else:
        raise ValueError("Invalid lander_type")

    return xi_arr, yi_arr, xy_arr


@njit(fastmath=True, nogil=True, cache=True)
def round2int(val: float) -> int:
    """Round to the nearest integer"""
    return int(round(val))


@njit(fastmath=True, nogil=True, cache=True)
def inside_line(x, y, x0, y0, x1, y1):
    """Return True if (x, y) is on the left of the line connecting from (x0, y0) to (x1, y1)"""
    v0x = x - x0
    v0y = y - y0
    v1x = x1 - x0
    v1y = y1 - y0
    # let v2 be cross product (v1 x v0)
    v2z = v1x * v0y - v1y * v0x
    return v2z > 0  # if True, (x,y) is located to the left of the line from (x0, y0) to (x1, y1)


@njit(fastmath=True, nogil=True, cache=True)
def cross_product(v1x, v1y, v1z, v2x, v2y, v2z):
    """Return cross product of v1 and v2"""
    v3x = v1y * v2z - v1z * v2y
    v3y = v1z * v2x - v1x * v2z
    v3z = v1x * v2y - v1y * v2x 
    return v3x, v3y, v3z


@njit(fastmath=True, nogil=True, cache=True)
def dot_product(v1x, v1y, v1z, v2x, v2y, v2z):
    """Return dot product of v1 and v2"""
    return v1x * v2x + v1y * v2y + v1z * v2z


@njit(fastmath=True, nogil=True, cache=True)
def point_closest2origin(x: float, y: float, res: float):
    """Return the closes point to the origin (0, 0) within the pixel located whose center is at (x, y) and has resolution res.
    
    Args:
        x (float): x coordinate of the pixel
        y (float): y coordinate of the pixel
        res (float): resolution of the pixel
        
    Returns:
        float: x coordinate of the closest point to the origin
        float: y coordinate of the closest point to the origin
    """

    theta = np.arctan2(-y, -x)
    #print(f"theta: {theta * 180 / np.pi} degrees")

    if -np.pi/4 <= theta <= np.pi/4:
        x_closest = res/2
        y_closest = np.tan(theta) * res/2
    elif np.pi/4 <= theta <= 3*np.pi/4:
        y_closest = res/2
        x_closest = res / 2 / np.tan(theta)
    elif -3*np.pi/4 <= theta <= -np.pi/4:
        y_closest = -res/2
        x_closest = res / 2 / np.tan(-theta)
    else:
        x_closest = -res/2
        y_closest = np.tan(np.pi - theta) * res/2

    return x + x_closest, y + y_closest


@njit(fastmath=True, nogil=True, cache=True)
def pixels_overlap_disk(diameter: float, res: float):
    """Create a disk mask; 1 if the pixel has overlap with the disk, 0 otherwise.
    The pixel diameter is set odd.
    
    Args:
        diameter (int): diameter of the disk.
        res (float): resolution of the rasterisation.
    
    Returns:
        np.array: rasterised disk mask.
    """
    s_diam = int(np.ceil(diameter / res))
    s_radius = int(s_diam // 2)

    s_mask = s_radius * 2 + 1
    mask = np.zeros((s_mask, s_mask))
    x_vec = np.linspace(-s_radius * res, s_radius * res, s_mask)
    y_vec = np.linspace(-s_radius * res, s_radius * res, s_mask)
    for i, x_ in enumerate(x_vec):
        for j, y_ in enumerate(y_vec):
            x, y = point_closest2origin(x_, y_, res)
            if x**2 + y**2 <= diameter**2 / 4:
                mask[i, j] = 1

    return mask, s_radius


def midpoint_circle(radius: int):
    """Return a list of points on the rasterized circle with the given radius. 
    The center of the circle is set (0, 0).
    Args:
        radius: radius of the circle (pix)

    Returns:
        circle_points: list of points on the circle, shape=(n, 2), dtype=int
    """

    circle_points = []

    cx, cy = 0, radius
    d = 1 - radius
    dH = 3
    dD = 5 - 2 * radius

    while cx <= cy:

        circle_points.append([
            [cx, cy],  # 0 - 45 degrees from (0, r)
            [cy, cx],  # 45 - 90 degrees
            [cy, -cx],  # 90 - 135 degrees
            [cx, -cy],  # 135 - 180 degrees
            [-cx, -cy],  # 180 - 225 degrees
            [-cy, -cx],  # 225 - 270 degrees
            [-cy, cx],  # 270 - 315 degrees
            [-cx, cy]  # 315 - 360 degrees
        ])

        if d < 0:
            d += dH
            dH += 2
            dD += 2
        else:
            d += dD
            dH += 2
            dD += 4
            cy -= 1

        cx += 1

    # sort data by angle
    circle_points = np.array(circle_points).reshape(-1, 2)
    theta = np.arctan2(circle_points[:, 1], circle_points[:, 0])
    circle_points = circle_points[np.argsort(theta)]

    # coordinate swap
    # Y                     +---------> Y (axis 1)
    # ^                     |
    # |                to   |
    # |                     |
    # + ---------> X        v X (axis 0)

    out = np.zeros_like(circle_points)
    out[:, 0] = -circle_points[:, 1]
    out[:, 1] = circle_points[:, 0]

    return out


@jit
def footprint_checker(lander_type: str, xi_arr: np.ndarray, yi_arr: np.ndarray, s_rlander: int):
    """Return the list of relative locations (pix) of the lander footprint

    Args:
        x_arr: array of x-coordinates of landing pads (m), shape (n, 4)
        y_arr: array of y-coordinates of landing pads (m), shape (n, 4)
        rmpp: map resolution in meter per pixel
        N_RLANDER: number of pixels to represent radius of lander

    Returns:
        within_footprint: array of boolean showing if the location is within the footprint, shape (n, m, m)
    """
    n = xi_arr.shape[0]
    m = 2 * s_rlander + 1

    within_footprint = np.zeros(shape=(n, m, m)).astype(np.bool_)

    for ti in range(n):
        left_xi, bottom_xi, right_xi, top_xi = xi_arr[ti]
        left_yi, bottom_yi, right_yi, top_yi = yi_arr[ti]

        for i in range(2 * s_rlander + 1):
            for j in range(2 * s_rlander + 1):
                xi = i - s_rlander
                yi = j - s_rlander

                cond1 = inside_line(xi, yi, top_xi, top_yi, left_xi, left_yi)
                cond2 = inside_line(xi, yi, left_xi, left_yi, bottom_xi, bottom_yi)
                cond3 = inside_line(xi, yi, bottom_xi, bottom_yi, right_xi, right_yi)
                cond4 = inside_line(xi, yi, right_xi, right_yi, top_xi, top_yi)
                cond5 = inside_line(xi, yi, right_xi, right_yi, left_xi, left_yi)

                if lander_type == "triangle":
                    within_footprint[ti, i, j] = cond2 and cond3 and cond5
                else:
                    within_footprint[ti, i, j] = cond1 and cond2 and cond3 and cond4

    return within_footprint

