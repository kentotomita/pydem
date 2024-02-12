from numba import jit
import numpy as np
from scipy.interpolate import LinearNDInterpolator


def pcd2dem(pcd: np.ndarray, res: float, method: str = "max"):
    """Convert point cloud data to Map object

    Args:
        pcd: shape = (# of points, 3=xyz)
        res: meter/pix for output DEM
        method: method for aggregating point cloud information into DEM grid. For example, if multiple point fall
            within a given cell, "max" will take the point with the maximum height.
    Returns:
        dem: height map
        pcd_idx_to_dem_coord: maps the index of each point in `pcd` to its corresponding coordinate in the DEM.
    """
    n, _ = np.shape(pcd)
    x = pcd[:, 0]
    y = pcd[:, 1]
    x0 = np.nanmin(x)
    xn = np.nanmax(x)
    y0 = np.nanmin(y)
    yn = np.nanmax(y)

    nx = int(np.ceil((xn - x0) / res))
    ny = int(np.ceil((yn - y0) / res))

    new_x = np.linspace(start=x0, stop=xn, num=nx)
    new_y = np.linspace(start=y0, stop=yn, num=ny)
    zz = np.zeros((nx, ny))
    h_sum = np.zeros((nx, ny))
    count = np.zeros((nx, ny))
    h_sum, count, pcd_idx_to_dem_coord = _pcd2dem(n, res, x0, y0, pcd, h_sum, count, method=method)
    # print(f"number of points: {n} | X range: {x0:.2f} to {xn:.2f} (m) | Y range: {y0:.2f} to {yn:.2f} (m) | Resolution {res} (m/pix)")
    # print(f"h_sum mean: {np.nanmean(np.abs(h_sum))} | h_sum max: {np.nanmax(np.abs(h_sum))}")
    print(f"count mean {np.nanmean(count)} | count max {np.nanmax(count)}")

    if method == "mean":
        zz[count > 0] = h_sum[count > 0] / count[count > 0]
    else:
        zz[count > 0] = h_sum[count > 0]
    zz[count == 0] = np.nan

    return zz, pcd_idx_to_dem_coord


#@jit
def _pcd2dem(n: int, res: float, x0, y0, pcd, h_sum, count, method: str):
    nx, ny = h_sum.shape
    flag = 0
    pcd_idx_to_dem_coord = {}
    for i in range(n):
        x = pcd[i, 0]
        y = pcd[i, 1]
        z = pcd[i, 2]

        xi = int((x - x0) // res)
        yi = int((y - y0) // res)

        if xi < nx and yi < ny:
            if method == "max":
                # print("before", z, h_sum[xi, yi], count[xi, yi])
                h_sum[xi, yi] = z if z > h_sum[xi, yi] or count[xi, yi] == 0 else h_sum[xi, yi]
                # print("after ", z, h_sum[xi, yi], count[xi, yi])
            elif method == "mean":
                h_sum[xi, yi] += z
            else:
                raise NameError(f"No method named {method}.")
            count[xi, yi] += 1
            pcd_idx_to_dem_coord[i] = (xi, yi)
        else:
            flag += 1
    if flag > 0:
        print(f"{flag} Out of bounds PCD entry detected")
    return h_sum, count, pcd_idx_to_dem_coord

