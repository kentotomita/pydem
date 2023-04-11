
import numpy as np
import matplotlib.pyplot as plt

import pydem.util as ut


if __name__=="__main__":
    # check 0-channel
    def make_grayimg(h, w):
        f = lambda x, y: np.sin(np.sqrt(x**2 + y**2))
        x = np.linspace(-6, 6, w)
        y = np.linspace(-6, 6, h)

        xx, yy = np.meshgrid(x, y)
        return f(xx, yy)
    
    img_before = make_grayimg(h=128, w=64)
    img_after = ut.interpolate_bilinear2d(img_before, 128, 64, np.zeros((64, 32)), 64, 32)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_before, cmap='gray')
    axs[1].imshow(img_after, cmap='gray')
    plt.show()

    # check 2-channel
    def make_rgbimg(h, w):
        f = lambda x, y, p1, p2: np.sin(np.sqrt(x**p1 + y**p2))
        x = np.linspace(-6, 6, w)
        y = np.linspace(-6, 6, h)
        xx, yy = np.meshgrid(x, y)
        r = f(xx, yy, 1, 2)
        g = f(xx, yy, 0, 1)
        b = f(xx, yy, 2, 3)
        out = np.zeros((h, w, 3))
        for i, c in enumerate([r, g, b]):
            out[:, :, i] = c
        return out
    
    img_before = make_rgbimg(h=128, w=64)
    img_after = ut.interpolate_bilinear3d(img_before, 128, 64, np.zeros((64, 32, 3)), 64, 32)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_before)
    axs[1].imshow(img_after)
    plt.show()


