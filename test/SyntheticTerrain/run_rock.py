"""testing rock generation algorithm"""
import matplotlib.pyplot as plt
import pydem.SyntheticTerrain as st

dem = st.rocky_terrain(shape=(128, 128), res=0.1, k=0.4, dmax=2., dmin=0.2)

plt.imshow(dem)
plt.show()
