# Hard dependencies ----------------------------------------------------------- #
# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "numba", 'scipy', 'matplotlib')

missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


# imports ---------------------------------------------------------------- #
__extensions__ = {'open3d': False, 'trimesh': False, 'pyvista': False}

# open3d
try:
    from open3d import __version__ as open3d_ver
    __extensions__['open3d'] = True
except ImportError:
    pass

# trimesh
try:
    from trimesh import __version__ as trimesh_ver
    __extensions__['trimesh'] = True
except ImportError:
    pass

# pyvista
try:
    from trimesh import __version__ as pyvista_ver
    __extensions__['pyvista'] = True
except ImportError:
    pass

# imports ---------------------------------------------------------------- #
# from . import Dataloader
from . import DataLoader
from . import SyntheticTerrain
from . import Geometry
from . import Graphic
from . import util
