from pydem import __extensions__

from .dem2stl import *
from ._util import *

if (__extensions__['open3d']):
    from .dem2mesh_o3d import *
    from .dem2boxmesh_o3d import *

if (__extensions__['trimesh']):
    from .dem2mesh_tri import *
    from .dem2boxmesh_tri import *
