"""Utility functions for Hazard Detection module."""

import numpy as np
from numba import jit, prange


def max_under_pad(n_rpad:int, dem:np.ndarray):
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
