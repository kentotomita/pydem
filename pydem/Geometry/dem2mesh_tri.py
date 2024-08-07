import trimesh
from ._util import _dem2mesh

def dem2mesh_tri(xx, yy, zz):
    """ Transform height map with x-y coordinate to 3D geometry data (Trimesh)

    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
    Return:
        mesh (trimesh.base.Trimesh): triangle mesh of DEM
    """
    vtc, tri = _dem2mesh(xx, yy, zz)

    # make 3d object
    mesh = trimesh.Trimesh(vertices=vtc, faces=tri)
    return mesh