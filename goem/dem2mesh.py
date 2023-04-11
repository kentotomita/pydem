"""2D --> 3D data tools"""

import numpy as np
from numba import cuda, jit
import open3d as o3d
import copy
import torch

def dem2mesh(xx, yy, zz, cuda=True):
    """ Transform height map with x-y coordinate to 3D geometry data (open3D)
    Ref: https://forum.open3d.org/t/add-triangle-to-empty-mesh/197/2

    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
    Return:
        mesh (o3d.geometry.TriangleMesh): triangle mesh of DEM
    """

    # instantiate arrays of vertices and triangles
    nr, nc = xx.shape
    vtc = np.zeros((nr*nc + (nr-1)*(nc-1), 3))
    tri = np.zeros(((nr-1)*(nc-1)*4, 3))

    # make/register vertices
    if cuda and torch.cuda.is_available():
        tpb = 512  # GPU computing config; threads_per_block
        bpg = 100  # GPU computing config; blocks_per_grid
        _make_vertices[bpg, tpb](xx, yy, zz, vtc)
    else:
        vtc = _make_vertices_cpu(xx, yy, zz, vtc)

    # register triangles
    tri = _make_triangles(nr, nc, tri)

    # make 3d object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vtc)
    mesh.triangles = o3d.utility.Vector3iVector(tri)
    return mesh

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


#@jit
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

