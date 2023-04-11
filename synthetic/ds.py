"""Diamond Square Algorithm (DSA)"""
import numpy as np
from numpy.random import RandomState
from numba import njit
from dem_tools.map.resize import grid_interpolate


def square_step(z, ns, w, l, rng):
    """Function computes square step (reference points form square).

    Args:
        z (np.ndarray): 2d array of initial height map but in refined resolution for adding small features
        ns (int): num of square grid that DSA has visited; initial step start with one and doubled every step
        w (int): edge width of each square grid; ns * w == len(z)-1
        l (float): limit of the added random value to the center point = mean of four corner points
        rng (RandomState): random number generator
    """
    noise = rng.uniform(-l, l, (ns, ns))
    z = _square_step(z, ns, w, noise)
    return z


@njit
def _square_step(z, ns, w, noise):
    """
    Args:
        noise(np.ndarray): shape=(ns, ns)
    """
    for i in range(ns):
        for j in range(ns):
            # REDEFINE STEP SIZE INCREMENTER & SHAPE INDICES.
            hw = w // 2          # half of grid width
            i_min = i * w        # dim0 index for left points of the grid
            i_max = (i + 1) * w  # dim0 index for right points of the grid
            j_min = j * w        # dim1 index for upper points of the grid
            j_max = (j + 1) * w  # dim1 index for lower points of the grid
            i_mid = i_min + hw   # dim0 index for the middle point of the grid
            j_mid = j_min + hw   # dim1 index for the middle point of the grid
            # ASSIGN REFERENCE POINTS & DO SQUARE STEP.
            ul = z[i_min, j_min]  # upper left
            ur = z[i_min, j_max]  # upper right
            ll = z[i_max, j_min]  # lower left
            lr = z[i_max, j_max]  # lower right
            z[i_mid, j_mid] = np.mean(np.array([ul, ur, ll, lr])) + noise[i, j]
    return z


def diamond_step(z, ns, w, l, rng):
    """Function computes diamond step (reference points form diamond).

    Args:
        z (np.ndarray): 2d array of initial height map but in refined resolution for adding small features
        ns (int): num of square grid that DSA has visited; initial step start with one and doubled every step
        w (int): edge width of each square grid; ns * w == len(z)-1
        l (float): limit of the added random value to the center point = mean of four corner points
        rng (RandomState): random number generator
    """
    mi = len(z)-1  # max index for corner case
    noise = rng.uniform(-l, l, (ns, ns, 4))
    z = _diamond_step(z, ns, w, mi, noise)

    return z


@njit
def _diamond_step(z, ns, w, mi, noise):
    """
    Args:
        noise(np.ndarray): shape=(ns,ns,4)
    """
    for i in range(ns):
        for j in range(ns):
            # REDEFINE STEP SIZE INCREMENTER & SHAPE INDICES.
            hw = w // 2             # half of grid width
            i_min = i * w           # dim0 index for left points of the grid
            i_max = (i + 1) * w     # dim0 index for right points of the grid
            j_min = j * w           # dim1 index for upper points of the grid
            j_max = (j + 1) * w     # dim1 index for lower points of the grid
            i_mid = i_min + hw      # dim0 index for the middle point of the grid
            j_mid = j_min + hw      # dim1 index for the middle point of the grid
            c = z[i_mid, j_mid]   # center
            ul = z[i_min, j_min]  # upper left
            ur = z[i_min, j_max]  # upper right
            ll = z[i_max, j_min]  # lower left
            lr = z[i_max, j_max]  # lower right
            # DO DIAMOND STEP.
            # Top Diamond
            tmp = mi - hw if i_min == 0 else i_min - hw  # wraps if at edge.
            if z[i_min, j_mid] == 0:  # If Top value exists then skip else compute.
                z[i_min, j_mid] = np.mean(np.array([c, ul, ur, z[tmp, j_mid]])) + noise[i, j, 0]

            # Left Diamond
            tmp = mi - hw if j_min == 0 else j_min - hw  # wraps if at edge.
            if z[i_mid, j_min] == 0:  # If Left value exists then skip else compute.
                z[i_min, j_mid] = np.mean(np.array([c, ul, ll, z[i_mid, tmp]])) + noise[i, j, 1]

            # Right Diamond
            tmp = 0 + hw if j_max == mi else j_max + hw  # wraps if at edge.
            z[i_mid, j_max] = np.mean(np.array([c, ur, lr, z[i_mid, tmp]])) + noise[i, j, 2]

            # Bottom Diamond
            tmp = 0 + hw if i_max == mi else i_max + hw  # wraps if at edge.
            z[i_max, j_mid] = np.mean(np.array([c, ll, lr, z[tmp, j_mid]])) + noise[i, j, 3]

    return z


def dsa(height_map: np.ndarray, steps: int=5, hmax: float=None, rng: RandomState=None):
    """Main looping function of diamond square algorithm.

    Args:
        height_map (np.ndarray): 2d array of initial height map but in refined resolution for adding small features
        steps (int): number of DS steps
        hmax (float): >0, height limit added (both in positive and negative directions)
        rng: seed
    """

    shape = height_map.shape
    assert shape[0] == shape[1]

    # resize height map
    new_shape = 2 ** steps + 1
    if shape != new_shape:
        height_map = grid_interpolate(height_map, size=new_shape, mode="bilinear")

    if hmax is None:
        hmax = np.min(height_map) + (np.max(height_map) - np.min(height_map)) * 1.1
    if rng is None:
        rng = RandomState(seed=0)

    # Set iterators
    w = len(height_map) - 1
    ns = 1  # Number of shapes is this number squared.
    for i in range(steps):
        lim = hmax / (i + 1)
        square_step(height_map, ns, w, lim, rng)
        diamond_step(height_map, ns, w, lim, rng)
        # Increment iterators for next loop. Use floor divide to force int.
        w //= 2
        ns *= 2

    # resize height map back
    if shape != new_shape:
        height_map = grid_interpolate(height_map, size=shape, mode="bilinear")

    return height_map
