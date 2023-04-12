"""Convert DEM to Image"""

import numpy as np
import matplotlib.pyplot as plt

def dem2rgbimg(dem: np.ndarray, cmap:str='viridis')->np.ndarray:
    """DEM of (H, W) to colored DEM of (H, W, C). Coordinate is managed by texture coordinates; no adjustment necessary here. 
    Args:
        dem: (h, w)
        cmap: matplotlib colormap

    References:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
        https://stackoverflow.com/questions/43457308/is-there-any-good-color-map-to-convert-gray-scale-image-to-colorful-ones-using-p
    """
    cm = plt.get_cmap(cmap)

    demmin = np.nanmin(dem)
    demmax = np.nanmax(dem)
    #demimg = np.flip(np.transpose((dem - demmin)/(demmax - demmin)), 0)
    #demimg = np.flip((dem - demmin)/(demmax - demmin), 0)
    demimg = (dem - demmin)/(demmax - demmin)
    demcolored = (cm(demimg) * 255).astype(np.uint8)
    return demcolored