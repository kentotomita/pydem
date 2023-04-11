import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def interpolate_bilinear2d(arr_in, h_in, w_in, arr_out, h_out, w_out):
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
            arr_out[i][j] = Dy_prev * (arr_in[y_next][x_prev] * Dx_next + arr_in[y_next][x_next] * Dx_prev) + Dy_next * (arr_in[y_prev][x_prev] * Dx_next + arr_in[y_prev][x_next] * Dx_prev)
                
    return arr_out


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def interpolate_bilinear3d(arr_in, h_in, w_in, arr_out, h_out, w_out):
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
            for c in prange(3):
                arr_out[i][j][c] = Dy_prev * (arr_in[y_next][x_prev][c] * Dx_next + arr_in[y_next][x_next][c] * Dx_prev) + Dy_next * (arr_in[y_prev][x_prev][c] * Dx_next + arr_in[y_prev][x_next][c] * Dx_prev)
                
    return arr_out

