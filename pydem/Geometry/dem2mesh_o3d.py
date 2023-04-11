import open3d as o3d

from ._util import _dem2mesh

def dem2mesh_o3d(xx, yy, zz, cuda=True):
    """ Transform height map with x-y coordinate to 3D geometry data (open3d)

    Args:
        xx (np.ndarray): x coordinate values in meshgrid manner
        yy (np.ndarray): y coordinate values in meshgrid manner
        zz (np.ndarray): height map
    Return:
        mesh (trimesh.base.Trimesh): triangle mesh of DEM
    """
    vtc, tri = _dem2mesh(xx, yy, zz, cuda)

    # make 3d object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vtc)
    mesh.triangles = o3d.utility.Vector3iVector(tri)
    return mesh