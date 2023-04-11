"""2D --> 3D data tools"""

from numba import cuda

@cuda.jit
def dem2stl(xx, yy, zz, tri):
    """Transform height map with x-y coordinate to STL data
    Ref: https://pypi.org/project/numpy-stl/
    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
        out (np.ndarray): output stl array; shape=(# of triangles, 3=vertices, 3=xyz)
    """
    xi0, yi0 = cuda.grid(2)
    dxi, dyi = cuda.gridsize(2)
    nr, nc = xx.shape

    for x0 in range(xi0, nr-1, dxi):
        for y0 in range(yi0, nc-1, dyi):
            # +---> Y
            # |
            # V
            # X
            #
            # You are at (0,0) and create a new point (1,1).
            # Then create the four triangles.
            #
            #(0,0)-------(0,2)
            #  |  \     /  |
            #  |   (1,1)   |
            #  |  /     \  |
            #(2,0)-------(2,2)
            #

            # take indices of neighbor pixels
            x2 = x0 + 1  # id of x-plus; south
            y2 = y0 + 1  # id of y-plus; east

            # --take x, y, z of neighbor pixels
            x00 = xx[x0, y0]
            y00 = yy[x0, y0]
            z00 = zz[x0, y0]

            x20 = xx[x2, y0]
            y20 = yy[x2, y0]
            z20 = zz[x2, y0]

            x02 = xx[x0, y2]
            y02 = yy[x0, y2]
            z02 = zz[x0, y2]

            x22 = xx[x2, y2]
            y22 = yy[x2, y2]
            z22 = zz[x2, y2]

            # create the middle point (1,1)
            x11 = (x00 + x20 + x02 + x22) / 4
            y11 = (y00 + y20 + y02 + y22) / 4
            z11 = (z00 + z20 + z02 + z22) / 4

            # Make four triangles--
            # point id
            pid = (x0*(nc-1) + y0)*4
            # north; 11-00-02
            tri[pid, 0, 0] = x11
            tri[pid, 0, 1] = y11
            tri[pid, 0, 2] = z11
            tri[pid, 1, 0] = x00
            tri[pid, 1, 1] = y00
            tri[pid, 1, 2] = z00
            tri[pid, 2, 0] = x02
            tri[pid, 2, 1] = y02
            tri[pid, 2, 2] = z02
            #out['vectors'][pid] = tri[pid, :, :]

            pid += 1
            # east; 11-02-22
            tri[pid, 0, 0] = x11
            tri[pid, 0, 1] = y11
            tri[pid, 0, 2] = z11
            tri[pid, 1, 0] = x02
            tri[pid, 1, 1] = y02
            tri[pid, 1, 2] = z02
            tri[pid, 2, 0] = x22
            tri[pid, 2, 1] = y22
            tri[pid, 2, 2] = z22
            #out['vectors'][pid] = tri[pid, :, :]

            pid += 1
            # south; 11-22-20
            tri[pid, 0, 0] = x11
            tri[pid, 0, 1] = y11
            tri[pid, 0, 2] = z11
            tri[pid, 1, 0] = x22
            tri[pid, 1, 1] = y22
            tri[pid, 1, 2] = z22
            tri[pid, 2, 0] = x20
            tri[pid, 2, 1] = y20
            tri[pid, 2, 2] = z20
            #out['vectors'][pid] = tri[pid, :, :]

            pid += 1
            # west; 11-20-00
            tri[pid, 0, 0] = x11
            tri[pid, 0, 1] = y11
            tri[pid, 0, 2] = z11
            tri[pid, 1, 0] = x20
            tri[pid, 1, 1] = y20
            tri[pid, 1, 2] = z20
            tri[pid, 2, 0] = x00
            tri[pid, 2, 1] = y00
            tri[pid, 2, 2] = z00
            #out['vectors'][pid] = tri[pid, :, :]








