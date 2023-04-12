"""Visualization using PyVista"""
import pyvista as pv
import numpy as np
try:
    import open3d as o3d
except:
    pass
try:
    import trimesh as tri
except:
    pass

from .demimg import dem2rgbimg

#def o3d2pv(mesh: o3d.geometry.TriangleMesh)->pv.PolyData:
#    """Convert Open3D TriangleMesh to PyVista PolyData"""
#    return pv.PolyData(var_inp=mesh.vertices, faces=mesh.triangles)


#def tri2pv(mesh: tri.Trimesh)->pv.PolyData:
#    """Convert Trimesh object to PyVista PolyData"""
#    return pv.PolyData(var_inp=mesh.vertices, faces=mesh.triangles)


def pvplot_dem(mesh:pv.PolyData, dem:np.ndarray, cmap='viridis'):
    """Visualize DEM in 3D, mapped with height."""

    # add texture coordinate
    mesh = demmesh_texture_coordinate(mesh)

    # convert dem to texture
    tex = dem2texture(dem, cmap=cmap)


    pl = pv.Plotter()
    pl.add_mesh(mesh, texture=tex)

    pl.add_axes()  # world axes arrows
    pl.show()


def dem2texture(dem:np.ndarray, cmap:str='viridis')->pv.Texture:
    """Create PyVista texture object from DEM"""
    demcolored = dem2rgbimg(dem, cmap)
    tex = pv.numpy_to_texture(demcolored)
    return tex


def demmesh_texture_coordinate(mesh: pv.PolyData):
    """Add texture coordinate to the DEM mesh. 
    PyVista mesh object is easily obtained by applying pv.wrap fuction; https://docs.pyvista.org/api/utilities/_autosummary/pyvista.wrap.html
    """

    # Create the ground control points for texture mapping
    pts = mesh.extract_surface().points  # pyvista_ndarray
    o = pts[:, 0].max(), pts[:, 1].min(), 0.0  # Bottom Left
    u = pts[:, 0].max(), pts[:, 1].max(), 0.0  # Bottom Right
    v = pts[:, 0].min(), pts[:, 1].min(), 0.0  # Lop left
    # Note: Z-coordinate doesn't matter
    # source: https://github.com/pyvista/pyvista-support/issues/159

    return mesh.texture_map_to_plane(o, u, v)



