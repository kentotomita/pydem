import numpy as np
from numba import jit, njit


def rocky_terrain(shape, res=0.1, k=0.4, dmax=2, dmin=0.2, rng=None):
    """
    Args:
        shape (tuple): shape of terrain array
        res(float): resolution (m/pix)
        k(float): coverage_rate; total area covered by rocks of all sizes
        dmax(float): maximum diameter of rocks
        dmin(float): minimum diameter of rocks
        rng(np.random.RandomState): random number generator
    """
    if rng is None:
        rng = np.random.RandomState(seed=0)

    nr, nc = shape
    area = (nr*res)*(nc*res)  # area of terrain (m^2)

    if k==0:
        terrain = np.zeros(shape=shape, dtype=np.float32)
    else:
        # get list of diameters and numbers of rocks
        ds, ns = _get_size_num(area, k, dmax, dmin)

        # number of pixels of maximum rock
        ndmax = np.ceil(dmax / res).astype(np.int32)

        # number of rocks to generate
        n_rock = np.sum(ns)

        # randomly generate their locations
        buffer = np.ceil(ndmax / 2).astype(np.int32) + 1
        loc0 = rng.uniform(buffer, nr - buffer, size=(n_rock,)).astype(np.int32)  # (rock id for axes0)
        loc1 = rng.uniform(buffer, nc - buffer, size=(n_rock,)).astype(np.int32)  # (rock id for axes1)
        loc = np.array([loc0, loc1]).transpose()

        # randomly generate rock parameters
        params = rng.uniform([1, 1.5, -np.pi], [1, 1.5, np.pi], size=(n_rock, 3))

        # generate rocks
        terrain = _place_rocks(shape=shape, res=res, d_list=ds, n_list=ns, loc=loc, params=params)

    return terrain


@jit
def _place_rocks(shape, res, d_list, n_list, loc, params):
    """
    Args:
        shape (tuple): shape of terrain array
        res(float): resolution (m/pix)
        d_list(np.ndarray): list of diameters (m)
        n_list(np.ndarray): list of numbers for each diameter
        loc(np.ndarray): shape=(nr, 2); [i, 0] and [i, 1] are x and y for rock i
        params(np.ndarray): shape=(nr, 3); [i, :] stores a, b, theta for rock i
    """
    # initialize terrain
    terrain = np.zeros(shape=shape, dtype=np.float32)

    # generate and locate rock
    rock_id = 0
    n_type = len(d_list)
    for j in range(n_type):
        d = d_list[j]
        num = n_list[j]
        for i in range(num):
            # generate rock
            a, b, theta = params[rock_id, :]
            rock, nrock = gen_rock(diameter=d, res=res, a=a, b=b, theta=theta)

            # locate rock
            ix, iy = loc[rock_id, :]
            nhrock = nrock // 2  # half size of pixel width
            ix0 = ix - nhrock
            ixn = ix0 + nrock
            iy0 = iy - nhrock
            iyn = iy0 + nrock
            terrain[ix0:ixn, iy0:iyn] += rock

            # update rock id
            rock_id += 1

    return terrain


@jit
def _get_size_num(area, k=0.4, dmax=2, dmin=0.2):
    """
    Args:
        area (float): area of terrain (m^2)
        k(float): coverage_rate; total area covered by rocks of all sizes
        dmax(float): maximum diameter of rocks
        dmin(float): minimum diameter of rocks
    Return:
        d_list(np.ndarray): list of diameters (m)
        n_list(np.ndarray): list of numbers for each diameter
    """

    # num of diameters to consider
    n_type = 5 * max(int(np.log10(dmax / dmin)), 1)

    # get list of diameters
    d_list = np.zeros(shape=(n_type,))
    for i in range(n_type):
        d_list[i] = dmax * 10 ** (-0.1*i)

    # list of cumulative area covered by rocks of a given diameter D [m] or larger
    f_list = np.zeros(shape=(n_type,))
    for i in range(n_type):
        f_list[i] = _F(k, d_list[i])

    # list of g(d); area covered by rocks whose diameter is d m
    g_list = np.zeros(shape=(n_type,))
    for i in range(n_type):
        if i == 0:
            g_list[0] = f_list[i]
        else:
            g_list[i] = f_list[i] - f_list[i - 1]

    # list of number of rocks for each sizes
    n_list = np.zeros(shape=(n_type,)).astype(np.int32)
    for i in range(n_type):
        n_list[i] = int(area * g_list[i] / (d_list[i] ** 2 * np.pi / 4))

    return d_list, n_list


@jit
def _F(k, D):
    """
    cumulative area covered by rocks of a given diameter D [m] or larger
    Args:
        D: Diameter
        k: coverage rate
    """
    if k == 0:
        f = 0
    else:
        q = 1.863 + 0.153 / k
        f = k * np.exp(-q * D)
    return f


@njit
def gen_rock(diameter, res, a=1, b=1, theta=np.pi/6):
    """generate simple rock
    Args:
        diameter(float): diameter of rock
        res(float): resolution (m/pix)
        a(float): ellipse coefficient of x coordinate, >=1
        b(float): ellipse coefficient of y coordinate, >=1
        theta(float): rotation angle (deg)
    Returns:
        z_grid(np.ndarray): heigh map of rock
        N(int): pixel size of the height map
    """
    R = diameter / 2
    N = max(int(diameter/res), 1)
    z_grid = np.zeros(shape=(N, N), dtype=np.float32)

    span = R
    x_vec = np.linspace(-span, span, N)
    y_vec = np.linspace(-span, span, N)
    for i in range(N):
        for j in range(N):
            x = x_vec[i]
            y = y_vec[j]
            r = np.sqrt(
                (a * (x * np.cos(theta) + y * np.sin(theta))) ** 2 +
                (b * (x * np.sin(theta) - y * np.cos(theta))) ** 2)  # assume roteted ellipse for the shape of the rock
            if r <= R:
                z_grid[j, i] = abs(R ** 2 - r ** 2) ** (1 / 2)
            else:
                z_grid[j, i] = 0
    return z_grid, N

