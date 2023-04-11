import numpy as np
import matplotlib.pyplot as plt

import pydem.SyntheticTerrain as st

if __name__=="__main__":
    import matplotlib.pyplot as plt
    dem_before = np.random.random(size=(128, 128))
    dem_after = st.dsa(dem_before)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dem_before)
    ax[0].title.set_text('DEM before')
    ax[1].imshow(dem_after)
    ax[1].title.set_text('DEM after')
    plt.show()
