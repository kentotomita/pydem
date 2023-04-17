import numpy as np
import sys
sys.path.append('../../')
import pydem.util as util

# Define the input image and kernel
image = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]])

kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])

# Perform dilation on the image using the kernel
dilated = util.dilation(image, kernel)
# Perform erosion on the image using the kernel
eroded = util.erosion(image, kernel)

# Print the dilated image
print(dilated)

# Print the eroded image
print(eroded)
