"""testing crater generation algorithm"""
from crater import gen_crater
import matplotlib.pyplot as plt

dem = gen_crater(d=50, res=0.1)

plt.imshow(dem)
plt.show()
