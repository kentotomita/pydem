"""Utility functions for mesh creations; vertices and triangles"""
import numpy as np
from numba import cuda, jit, prange
import copy


# =================================================================
# Functions for mesh creation from DEM
# =================================================================

def _dem2mesh(xx, yy, zz, cuda=False):
    """ Transform height map with x-y coordinate to 3D geometry data; vertices and triangles

    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
    Return:
        vtc: vertices
        tri: triangles
    """

    # instantiate arrays of vertices and triangles
    nr, nc = xx.shape
    vtc = np.zeros((nr*nc + (nr-1)*(nc-1), 3))
    tri = np.zeros(((nr-1)*(nc-1)*4, 3))

    # make/register vertices
    if cuda:
        try:
            tpb = 512  # GPU computing config; threads_per_block
            bpg = 100  # GPU computing config; blocks_per_grid
            _make_vertices[bpg, tpb](xx, yy, zz, vtc)
        except:
            print("Couldn't run cuda version, shifted to cpu version.")
            vtc = _make_vertices_cpu(xx, yy, zz, vtc)
    else:
        vtc = _make_vertices_cpu(xx, yy, zz, vtc)

    # register triangles
    tri = _make_triangles(nr, nc, tri)
    return vtc, tri


@jit
def _make_triangles(nr, nc, tri):
    """
    Args:
        nr (int): # of rows
        nc (int): # of columns
        tri (np.ndarray.astype(np.int32)): empty triangle array to be filled, shape=(# of triangles, 3)
    """
    # ============================
    # Make triangles
    # ============================
    # Numbering system
    # (0)-----(1)-----(2)
    #  | \   / | \   / |
    #  |  (3)  |  (4)  |
    #  | /   \ | /   \ |
    # (5)-----(6)-----(7)
    #  | \   / | \   / |
    #  |  (8)  |  (9)  |
    #  | /   \ | /   \ |
    for i in range(nr-1):
        for j in range(nc-1):
            # suppose are at the up left point (e.g., 0, 1, 5, 6, in the fig above)
            # let this point be the origin (0,0)
            # (0,0)-------(0,2)
            #  |  \     /  |
            #  |   (1,1)   |
            #  |  /     \  |
            # (2,0)-------(2,2)

            # vertex indices
            id00 = i * (nc + nc-1) + j
            id02 = id00 + 1
            id11 = id00 + nc
            id20 = (i+1) * (nc + nc-1) + j
            id22 = id20 + 1

            # triangle index; four triangles per "up left" point
            tid = (i * (nc-1) + j)*4

            # register triangles
            # 00-02-11
            tri[tid, 0] = id00
            tri[tid, 1] = id02
            tri[tid, 2] = id11
            # 02-22-11
            tid += 1
            tri[tid, 0] = id02
            tri[tid, 1] = id22
            tri[tid, 2] = id11
            # 22-20-11
            tid += 1
            tri[tid, 0] = id22
            tri[tid, 1] = id20
            tri[tid, 2] = id11
            # 20-00-11
            tid += 1
            tri[tid, 0] = id20
            tri[tid, 1] = id00
            tri[tid, 2] = id11
    return tri


@cuda.jit
def _make_vertices(xx, yy, zz, vtc):
    """
    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
        vtc (np.ndarray): vertices, shape=(# of vertices, 3=xyz)
    """
    xi0, yi0 = cuda.grid(2)
    dxi, dyi = cuda.gridsize(2)
    nr, nc = xx.shape

    for x0 in range(xi0, nr, dxi):
        for y0 in range(yi0, nc, dyi):
            # +---> Y
            # |
            # V
            # X

            #============================
            # register current point
            #============================
            # Numbering system; here (3), (4), (8), (9),.. are newly created vertices
            # (0)-----(1)-----(2)
            #  | \   / | \   / |
            #  |  (3)  |  (4)  |
            #  | /   \ | /   \ |
            # (5)-----(6)-----(7)
            #  | \   / | \   / |
            #  |  (8)  |  (9)  |
            #  | /   \ | /   \ |

            # current coordinate
            x00 = xx[x0, y0]
            y00 = yy[x0, y0]
            z00 = zz[x0, y0]
            # vertex id
            vid = x0 * (nc + nc - 1) + y0
            # register current coordinate
            vtc[vid, 0] = x00
            vtc[vid, 1] = y00
            vtc[vid, 2] = z00

            # ============================
            # create middle point
            # ============================
            # You are at (0,0) and create a new point (1,1).
            # Then create the four triangles.
            #
            #(0,0)-------(0,2)
            #  |  \     /  |
            #  |   (1,1)   |
            #  |  /     \  |
            #(2,0)-------(2,2)
            #
            if x0 < nr-1 and y0 < nc-1:
                # take indices of neighbor pixels
                x2 = x0 + 1  # id of x-plus; south
                y2 = y0 + 1  # id of y-plus; east

                # --take x, y, z of neighbor pixels
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

                # ============================
                # register middle point
                # ============================
                # vertex id
                vid = x0 * (nc + nc - 1) + y0 + nc
                # register middle point coordinate
                vtc[vid, 0] = x11
                vtc[vid, 1] = y11
                vtc[vid, 2] = z11



@jit
def _make_vertices_cpu(xx, yy, zz, vtc):
    """
    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
        vtc (np.ndarray): vertices, shape=(# of vertices, 3=xyz)
    """
    nr, nc = xx.shape

    for x0 in range(nr):
        for y0 in range(nc):
            # +---> Y
            # |
            # V
            # X

            #============================
            # register current point
            #============================
            # Numbering system; here (3), (4), (8), (9),.. are newly created vertices
            # (0)-----(1)-----(2)
            #  | \   / | \   / |
            #  |  (3)  |  (4)  |
            #  | /   \ | /   \ |
            # (5)-----(6)-----(7)
            #  | \   / | \   / |
            #  |  (8)  |  (9)  |
            #  | /   \ | /   \ |

            # current coordinate
            x00 = xx[x0, y0]
            y00 = yy[x0, y0]
            z00 = zz[x0, y0]
            # vertex id
            vid = x0 * (nc + nc - 1) + y0
            # register current coordinate
            vtc[vid, 0] = x00
            vtc[vid, 1] = y00
            vtc[vid, 2] = z00

            # ============================
            # create middle point
            # ============================
            # You are at (0,0) and create a new point (1,1).
            # Then create the four triangles.
            #
            #(0,0)-------(0,2)
            #  |  \     /  |
            #  |   (1,1)   |
            #  |  /     \  |
            #(2,0)-------(2,2)
            #
            if x0 < nr-1 and y0 < nc-1:
                # take indices of neighbor pixels
                x2 = x0 + 1  # id of x-plus; south
                y2 = y0 + 1  # id of y-plus; east

                # --take x, y, z of neighbor pixels
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

                # ============================
                # register middle point
                # ============================
                # vertex id
                vid = x0 * (nc + nc - 1) + y0 + nc
                # register middle point coordinate
                vtc[vid, 0] = x11
                vtc[vid, 1] = y11
                vtc[vid, 2] = z11
    return copy.deepcopy(vtc)


# =================================================================
# Functions for box-mesh (something like voxel) creation from DEM
# =================================================================


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _make_boxtriangles(nr, nc, tri):
    """
    Args:
        nr (int): # of rows
        nc (int): # of columns
        tri (np.ndarray.astype(np.int32)): empty triangle array to be filled, shape=(# of triangles, 3)
    """
    # ============================
    # Make triangles
    # ============================
    # Numbering system
    # (0)-(1)-(2)
    #  | \ | \ | 
    # (3)-(4)-(5)
    #  | \ |
    # (6)-(7)
    for i in prange(nr-1):
        for j in prange(nc-1):
            # suppose are at the up left point (e.g., 0, 1, 5, 6, in the fig above)
            # let this point be the origin (0,0)
            # (0,0)-------(0,2)
            #  |  \     /  |
            #  |   (1,1)   |
            #  |  /     \  |
            # (2,0)-------(2,2)

            # vertex indices
            id00 = i * (nc + nc-1) + j
            id02 = id00 + 1
            id11 = id00 + nc
            id20 = (i+1) * (nc + nc-1) + j
            id22 = id20 + 1

            # triangle index; four triangles per "up left" point
            tid = (i * (nc-1) + j)*4

            # register triangles
            # 00-02-11
            tri[tid, 0] = id00
            tri[tid, 1] = id02
            tri[tid, 2] = id11
            # 02-22-11
            tid += 1
            tri[tid, 0] = id02
            tri[tid, 1] = id22
            tri[tid, 2] = id11
            # 22-20-11
            tid += 1
            tri[tid, 0] = id22
            tri[tid, 1] = id20
            tri[tid, 2] = id11
            # 20-00-11
            tid += 1
            tri[tid, 0] = id20
            tri[tid, 1] = id00
            tri[tid, 2] = id11
    return tri



@jit
def _make_boxvertices(xx, yy, zz, vtc):
    """
    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
        vtc (np.ndarray): vertices, shape=(# of vertices, 3=xyz)
    """
    nr, nc = xx.shape

    for xi in range(nr):
        for yi in range(nc):
            # +---> Y
            # |
            # V
            # X

            #============================
            # register current point
            #============================
            # Numbering system; (A), (B), ... are data points, and (0), (1), ... are viertices to be crated
            # (0)-----(1)-----(4)-----(5)-----(8)
            #  | \     | \     | \     | \     |
            #  |  (A)  |   \   |  (B)  |   \   |...
            #  |     \ |     \ |     \ |     \ |
            # (2)-----(3)-----(6)-----(7)-----(9)
            #  | \     |       | \     |
            #  |   \   |       |   \   |
            #  |     \ |       |     \ |
            # (12)- --(13)----(16)----(17)----(8)
            #  | \     | \     | \     | \     |
            #  |  (E)  |   \   |  (F)  |   \   |...
            #  |     \ |     \ |     \ |     \ |
            # (14)----(15)----(18)----(19)----(9)
            # 

            # -----------------------------------------
            # Extract coordinates of the data points
            # You are at (1,1) 
            #
            #       (0,1)
            #         |
            #(1,0)--(1,1)--(1,2)
            #         |
            #       (2,1)
            #

            # current coordinate; (1,1)
            x11 = xx[xi, yi]
            y11 = yy[xi, yi]
            z11 = zz[xi, yi]
            # top; (0,1)
            if xi > 0:
                x01 = xx[xi-1, yi]
                y01 = yy[xi-1, yi]
                z01 = zz[xi-1, yi]
            # right; (1,2)
            if yi < nc:
                x12 = xx[xi, yi+1]
                y12 = yy[xi, yi+1]
                z12 = zz[xi, yi+1]
            # bottom; (2,1)
            if xi < nr:
                x21 = xx[xi+1, yi]
                y21 = yy[xi+1, yi]
                z21 = zz[xi+1, yi]
            # left; (1,0)
            if xi > 0:
                x10 = xx[xi, yi-1]
                y10 = yy[xi, yi-1]
                z10 = zz[xi, yi-1]

            # -----------------------------------------
            # Define the vertices; you are at (X)
            # (a)-------(b)
            #  | \     / |
            #  |   (X)   |
            #  | /     \ |
            # (c)-------(d)

            if 0 < xi < nr:
                xab = (x11 + x01) / 2
                xcd = (x11 + x21) / 2
            elif xi==0:
                xab = x11 - (x21 - x11)/2
                xcd = (x11 + x21) / 2
            else:  # xi==nr
                xab = (x11 + x01) / 2
                xcd = x11 + (x11 - x01)/2

            if 0 < yi < nc:
                yac = (y11 + y10) / 2
                ybd = (y11 + y12) / 2
            elif yi==0:
                yac = y11 - (y12 - y11) / 2
                ybd = (y11 + y12) / 2
            else:  # yi==nc
                yac = (y11 + y10) / 2
                ybd = y11 + (y11 - y10) / 2

            # vertex id for (a)
            vid = xi * nc * 4 + yi * 4

            vtc[vid, :] = xab, yac, z11      # register (a)
            vtc[vid + 1, :] = xab, ybd, z11  # register (b)
            vtc[vid + 2, :] = xcd, yac, z11  # register (c)
            vtc[vid + 3, :] = xcd, ybd, z11  # register (d)
            
    return copy.deepcopy(vtc)
