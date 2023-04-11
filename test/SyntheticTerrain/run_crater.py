"""testing crater generation algorithm"""

import sys
sys.path.append("../../")
import pydem.SyntheticTerrain as st
import matplotlib.pyplot as plt

dem = st.gen_crater(d=50, res=0.1)

plt.imshow(dem)
plt.show()
