"""Utility functions for mesh creations; vertices and triangles"""
import numpy as np
from numba import jit, prange
import copy


# =================================================================
# Functions for mesh creation from DEM
# =================================================================

def _dem2mesh(xx, yy, zz):
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
    vtc = _make_vertices(nr, nc, xx, yy, zz, vtc)

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
            # 00-11-02
            tri[tid, :] = id00, id11, id02
            # 02-11-22
            tid += 1
            tri[tid, :] = id02, id11, id22
            # 22-11-20
            tid += 1
            tri[tid, :] = id22, id11, id20
            # 20-11-00
            tid += 1
            tri[tid, :] = id20, id11, id00
    return tri




@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _make_vertices(nr, nc, xx, yy, zz, vtc):
    """
    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
        vtc (np.ndarray): vertices, shape=(# of vertices, 3=xyz)
    """

    for x0 in prange(nr):
        for y0 in prange(nc):
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
    #return copy.deepcopy(vtc)
    return vtc


# =================================================================
# Functions for box-mesh (something like voxel) creation from DEM
# =================================================================

def _dem2boxmesh(xx, yy, zz):
    """ Transform height map with x-y coordinate to voxel-type 3D geometry data; vertices and triangles

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
    vtc = np.zeros((nr*nc*4, 3))
    tri = np.zeros(((nr-1)*(nc-1)*6 + (nr-1)*4 + (nc-1)*4 + 2, 3))

    # make/register vertices
    vtc = _make_boxvertices(nr, nc, xx, yy, zz, vtc)

    # register triangles
    tri = _make_boxtriangles(nr, nc, tri)
    return vtc, tri


@jit
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
    # (0)-(1)-(5)
    #  | \ | \ | 
    # (3)-(4)-(5)
    #  | \ |
    # (6)-(7)
    tid = 0
    for i in prange(nr):
        for j in prange(nc):
            # +---> Y
            # |
            # V
            # X

            #============================
            # register current point
            #============================
            # Numbering system; (A), (B), ... are data points, and (0), (1), ... are viertices crated
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

            # For each data point (X), we create the following 6 triangles (except edges)
            # (0)-----(1)-----(4)
            #  | \     | \     |
            #  |  (X)  |   \   |
            #  |     \ |     \ |
            # (2)-----(3)-----(6)
            #  | \     |        
            #  |   \   |        
            #  |     \ |          
            # (12)----(13)-
            
            # let's denote them by the fllowing
            # (a)-----(b)-----(c)
            #  | \     | \     |
            #  |  (X)  |   \   |
            #  |     \ |     \ |
            # (d)-----(e)-----(f)
            #  | \     |        
            #  |   \   |        
            #  |     \ |          
            # (g)-----(h)-

            # vertex indices
            ida = i * nc * 4 + j * 4
            idb = ida + 1
            idd = ida + 2
            ide = ida + 3
            idc = ide + 1
            idf = idc + 2
            idg = ida + nc * 4
            idh = idg + 1

            # triangle index; 6 triangles per data point
            #tid = i * (nc-2) * 6 + i * 2 + j * 6

            # register triangles
            # a-e-b
            tri[tid, :] = ida, ide, idb
            tid += 1
            # a-d-e
            tri[tid, :] = ida, idd, ide
            tid += 1
            if j < nc-1:
                # b-f-c
                tri[tid, :] = idb, idf, idc
                tid += 1
                # b, e, f
                tri[tid, :] = idb, ide, idf
                tid += 1
            if i < nr-1:
                # d-h-e
                tri[tid, :] = idd, idh, ide
                tid += 1
                # d-g-h
                tri[tid, :] = idd, idg, idh
                tid += 1
    return tri



@jit
def _make_boxvertices(nr, nc, xx, yy, zz, vtc):
    """
    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
        vtc (np.ndarray): vertices, shape=(# of vertices, 3=xyz)
    """
    #nr, nc = xx.shape
    vid = 0

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
            if yi < nc-1:
                x12 = xx[xi, yi+1]
                y12 = yy[xi, yi+1]
                z12 = zz[xi, yi+1]
            # bottom; (2,1)
            if xi < nr-1:
                x21 = xx[xi+1, yi]
                y21 = yy[xi+1, yi]
                z21 = zz[xi+1, yi]
            # left; (1,0)
            if yi > 0:
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

            if 0 < xi < nr-1:
                xab = (x11 + x01) / 2
                xcd = (x11 + x21) / 2
            elif xi==0:
                xab = x11 - (x21 - x11)/2
                xcd = (x11 + x21) / 2
            else:  # xi==nr-1
                xab = (x11 + x01) / 2
                xcd = x11 + (x11 - x01)/2

            if 0 < yi < nc-1:
                yac = (y11 + y10) / 2
                ybd = (y11 + y12) / 2
            elif yi==0:
                yac = y11 - (y12 - y11) / 2
                ybd = (y11 + y12) / 2
            else:  # yi==nc-1
                yac = (y11 + y10) / 2
                ybd = y11 + (y11 - y10) / 2

            # vertex id for (a)
            #vid = xi * nc * 4 + yi * 4

            # register vertices
            # (a)
            vtc[vid, 0] = xab
            vtc[vid, 1] = yac
            vtc[vid, 2] = z11
            vid += 1
            # (b)
            vtc[vid, 0] = xab
            vtc[vid, 1] = ybd
            vtc[vid, 2] = z11
            vid += 1
            # (c)
            vtc[vid, 0] = xcd
            vtc[vid, 1] = yac
            vtc[vid, 2] = z11   
            vid += 1
            # (d)
            vtc[vid, 0] = xcd
            vtc[vid, 1] = ybd
            vtc[vid, 2] = z11
            vid += 1

            """
            vtc[vid, :] = xab, yac, z11  # register (a)
            vid += 1
            vtc[vid, :] = xab, ybd, z11  # register (b)
            vid += 1
            vtc[vid, :] = xcd, yac, z11  # register (c)
            vid += 1
            vtc[vid, :] = xcd, ybd, z11  # register (d)
            vid += 1
            """
            
    return vtc
