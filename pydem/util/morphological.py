import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def dilation(image, kernel):
    """Perform dilation on a binary image using a structuring element (kernel).

    Args:
        image (ndarray): 2D binary image as a numpy array.
        kernel (ndarray): 2D structuring element as a numpy array.

    Returns:
        ndarray: Dilated image as a numpy array.
    """
    # Get the dimensions of the image and kernel
    rows, cols = image.shape
    krows, kcols = kernel.shape

    # Initialize the output image
    dilated = np.zeros_like(image)

    # Iterate over each pixel in the image
    for i in prange(rows):
        for j in prange(cols):
            # Check if the kernel overlaps with white pixels in the image
            overlap = False
            for k in prange(krows):
                for l in prange(kcols):
                    row = i - krows // 2 + k
                    col = j - kcols // 2 + l
                    if row < 0 or row >= rows or col < 0 or col >= cols:
                        continue
                    if kernel[k, l] and image[row, col]:
                        overlap = True
                        break
                if overlap:
                    break
            
            if overlap:
                # Set the output pixel to white (1)
                dilated[i, j] = 1

    return dilated



@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def erosion(image, kernel):
    """Perform erosion on a binary image using a structuring element (kernel).

    Args:
        image (ndarray): 2D binary image as a numpy array.
        kernel (ndarray): 2D structuring element as a numpy array.

    Returns:
        ndarray: Eroded image as a numpy array.
    """
    # Get the dimensions of the image and kernel
    rows, cols = image.shape
    krows, kcols = kernel.shape

    # Initialize the output image
    eroded = np.zeros_like(image)

    # Iterate over each pixel in the image
    for i in prange(rows):
        for j in prange(cols):
            # Check if the kernel overlaps with white pixels in the image
            overlap = True
            for k in prange(krows):
                for l in prange(kcols):
                    row = i - krows // 2 + k
                    col = j - kcols // 2 + l
                    if row < 0 or row >= rows or col < 0 or col >= cols:
                        continue
                    if kernel[k, l] and not image[row, col]:
                        overlap = False
                        break
                if not overlap:
                    break
            
            if overlap:
                # Set the output pixel to white (1)
                eroded[i, j] = 1

    return eroded