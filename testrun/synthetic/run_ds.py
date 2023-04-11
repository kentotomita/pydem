"""testing diamond square algorithm"""
import numpy as np
import matplotlib.pyplot as plt

import ds

dem_before = np.random.random(size=(128, 128))
dem_after = dsa(dem_before)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(dem_before)
ax[1].imshow(dem_after)
plt.show()

