

import trimesh
from ._util import _dem2boxmesh

def dem2boxmesh_tri(xx, yy, zz):
    """ Transform height map with x-y coordinate to voxel-type 3D geometry data (Trimesh)

    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
    Return:
        mesh (trimesh.base.Trimesh): voxel-type triangle mesh of DEM
    """
    vtc, tri = _dem2boxmesh(xx, yy, zz)

    # make 3d object
    mesh = trimesh.Trimesh(vertices=vtc, faces=tri)
    return mesh


