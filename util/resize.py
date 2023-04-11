import numpy as np
from numba import jit, prange

#@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def interpolate_bilinear(arr_in, h_in, w_in, c_in, arr_out, h_out, w_out):
    """bilinear interpolation
    Args:
        arr_in: shape = (h_in, w_in, c_in)

    Source: https://eng.aurelienpierre.com/2020/03/bilinear-interpolation-on-images-stored-as-python-numpy-ndarray/
    """
    for i in prange(h_out):
        for j in prange(w_out):
            # Relative coordinates of the pixel in output space
            x_out = j / w_out
            y_out = i / h_out

            # Corresponding absolute coordinates of the pixel in input space
            x_in = (x_out * w_in)
            y_in = (y_out * h_in)

            # Nearest neighbours coordinates in input space
            x_prev = int(np.floor(x_in))
            x_next = x_prev + 1
            y_prev = int(np.floor(y_in))
            y_next = y_prev + 1

            # Sanitize bounds - no need to check for < 0
            x_prev = min(x_prev, w_in - 1)
            x_next = min(x_next, w_in - 1)
            y_prev = min(y_prev, h_in - 1)
            y_next = min(y_next, h_in - 1)
            
            # Distances between neighbour nodes in input space
            Dy_next = y_next - y_in;
            Dy_prev = 1. - Dy_next; # because next - prev = 1
            Dx_next = x_next - x_in;
            Dx_prev = 1. - Dx_next; # because next - prev = 1
            
            # Interpolate over channels
            if c_in == 1:
                arr_out[i][j] = Dy_prev * (arr_in[y_next][x_prev] * Dx_next + arr_in[y_next][x_next] * Dx_prev) + Dy_next * (arr_in[y_prev][x_prev] * Dx_next + arr_in[y_prev][x_next] * Dx_prev)
            else:
                for c in prange(c_in):
                    arr_out[i][j][c] = Dy_prev * (arr_in[y_next][x_prev][c] * Dx_next + arr_in[y_next][x_next][c] * Dx_prev) + Dy_next * (arr_in[y_prev][x_prev][c] * Dx_next + arr_in[y_prev][x_next][c] * Dx_prev)
                
    return arr_out


if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # check 1-channel
    def make_grayimg(h, w):
        f = lambda x, y: np.sin(np.sqrt(x**2 + y**2))
        x = np.linspace(-6, 6, w)
        y = np.linspace(-6, 6, h)

        xx, yy = np.meshgrid(x, y)
        return f(xx, yy)
    
    img_before = make_grayimg(h=128, w=64)
    img_after = interpolate_bilinear(img_before, 128, 64, 1, np.zeros((64, 32)), 64, 32)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_before, cmap='gray')
    axs[1].imshow(img_after, cmap='gray')
    plt.show()

    # check 3-channel
    def make_rgbimg(h, w):
        f = lambda x, y, p1, p2: np.sin(np.sqrt(x**p1 + y**p2))
        x = np.linspace(-6, 6, w)
        y = np.linspace(-6, 6, h)
        xx, yy = np.meshgrid(x, y)
        r = f(xx, yy, 2, 2)
        g = f(xx, yy, 1, 1)
        b = f(xx, yy, 3, 3)
        out = np.zeros((h, w, 3))
        for i, c in enumerate([r, g, b]):
            out[:, :, i] = c
        return out
    
    img_before = make_rgbimg(h=128, w=64)
    img_after = interpolate_bilinear(img_before, 128, 64, 3, np.zeros((64, 32, 3)), 64, 32)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_before)
    axs[1].imshow(img_after)
    plt.show()



