import numpy as np
import os
import matplotlib.pyplot as plt

OBJECT_IND = 2   # length of indent in OBJECT in the header
IMAGE_VAL = 21   # length before the value in OBJECT=IMAGE in the header
IMP_VAL = 33     # length before the value in OBJECT=IMAGE_MAP_PROJECTION in the header

class HiriseDtm():
    """
    HiRISE DTM source: https://www.uahirise.org/dtm/index.php?page=2
    Details of data can be read in [file name]_header.txt, which is automatically generated at instantiation. Some
    important description is shown below:

    'Pixel values in this file represent elevations in
    meters above the martian equipotential surface (Mars
    2000 Datum) defined by Smith, et al. (2001). Conversion
    from pixel units to geophysical units is given by the
    keyvalues for SCALING_FACTOR and OFFSET. This DTM was
    produced using ISIS and SOCET Set (copyright BAE
    Systems) software as described in Kirk et al. (2008).'

    Attributes:
        elevation(np.ndarray): height data in grid (m)
        x_grid(np.ndarray): x data in grid (m)
        y_grid(np.ndarray): y data in grid (m)
        img_dir(str): path to the data directory
        img_name(str): data file name
        img_path(str): path to the data file
        OBJECT_IND(int): length of indent in OBJECT in the header
        LINES(int): number of line scans of data
        LINE_SAMPLES(int): number of samples per line-scan
        OFFSET(float): offset for conversion from pixel units to geophysical units
        SCALING_FACTOR(float): scaling factor for conversion from pixel units to geophysical units
        VALID_MINIMUM(float): valid minimum elevation (m)
        VALID_MAXIMUM(float): valid maximum elevation (m)
        MAP_SCALE(float): data resolution (meters/pixel)

    -- Usage --
    HiriseDtm.img -> access image after post processing
    HiriseDtm.raw_img -> access image before post processing
    HiriseDtm.img_shape -> access shape of image
    HiriseDtm.visualize() -> visualize processed image

    Array view (axis=0: X, axis=1: Y)
        + ----> Y   |
        |           | Use meshgrid in this order!
        V           | yy, xx = np.meshgrid(y, x)
        X           |
    """
    def __init__(self, img_dir, img_name):
        self.img_dir = img_dir
        self.img_name = img_name
        self.img_path = os.path.join(img_dir, img_name)
        self.OBJECT_IND = OBJECT_IND
        self.LINES = 0
        self.LINE_SAMPLES = 0
        self.OFFSET = 0
        self.SCALING_FACTOR = 0
        self.VALID_MINIMUM = 0
        self.VALID_MAXIMUM = 0
        self.MAP_SCALE = 0
        self._read_header()
        self._read_image()

    def _read_header(self):
        f = open(self.img_path, 'rt')
        count = 0
        line = ''
        header_path = os.path.join(self.img_dir, self.img_name[:-4] + '_header.txt')
        header = open(header_path, 'w')
        while line != 'END\n' and count <= 300:
            line = f.readline()
            self.LINES          = self.__read_header_value(self.LINES, line, 'LINES', int, len_skip=IMAGE_VAL)
            self.LINE_SAMPLES   = self.__read_header_value(self.LINE_SAMPLES, line, 'LINE_SAMPLES', int, len_skip=IMAGE_VAL)
            self.OFFSET         = self.__read_header_value(self.OFFSET, line, 'OFFSET', np.float32, len_skip=IMAGE_VAL)
            self.SCALING_FACTOR = self.__read_header_value(self.SCALING_FACTOR, line, 'SCALING_FACTOR', np.float32, len_skip=IMAGE_VAL)
            self.VALID_MINIMUM  = self.__read_header_value(self.VALID_MINIMUM, line, 'VALID_MINIMUM', np.float32, len_skip=IMAGE_VAL)
            self.VALID_MAXIMUM  = self.__read_header_value(self.VALID_MAXIMUM, line, 'VALID_MAXIMUM', np.float32, len_skip=IMAGE_VAL)
            self.MAP_SCALE      = self.__read_header_value(self.MAP_SCALE, line, 'MAP_SCALE', np.float32, len_skip=IMP_VAL, figure2unit=3)
            header.write(line)
            count += 1
            if count == 300: print('Warning: header is larger than pre-defined limit')
        header.close()
        f.close()

    def __read_header_value(self, var, line, name, dtype, len_skip, figure2unit=0):
        if line[self.OBJECT_IND : self.OBJECT_IND + len(name)] == name:
            if figure2unit == 0:
                return np.array(line[len_skip:], dtype=dtype).tolist()
            else:
                return np.array(line[len_skip : len_skip + figure2unit], dtype=dtype).tolist()
        else:
            return var

    def _read_image(self):
        """
        Upper left corner of the image, or [0, 0] element of the data array, is set to the origin (x,y)=(0,0).
        Altitude is set positive.

        Array view (axis=0: X, axis=1: Y)
        + ----> Y   |
        |           | Use meshgrid in this order!
        V           | yy, xx = np.meshgrid(y, x)
        X           |
        """
        shape = (self.LINES, self.LINE_SAMPLES)  # matrix size
        n_points = shape[0] * shape[1]
        f = open(self.img_path, 'rb')
        data = np.fromfile(f, dtype=np.float32)
        f.close()
        data = data[len(data) - n_points:]
        data = self.SCALING_FACTOR * data
        data = self.OFFSET + data
        # data = self.OFFSET + self.SCALING_FACTOR * data   # Got segmentation error on PACE
        data = data.reshape(shape)
        data[data < self.VALID_MINIMUM] = np.nan
        data[data > self.VALID_MAXIMUM] = np.nan

        # elevation (z) in meter
        self.elevation = data.copy()

        # x, y coordinate of each pixels as sparse output of meshgrid
        h, w = shape  # hight: x-axis, width: y-axis
        self.x = np.linspace(0, (h-1), h).astype(np.float32) * self.MAP_SCALE
        self.y = np.linspace(0, (w-1), w).astype(np.float32) * self.MAP_SCALE

        self.shape = shape

    def prep_grid(self):
        y_grid, x_grid = np.meshgrid(self.y, self.x)
        return x_grid, y_grid

    def visualize(self):
        z = np.ma.array(self.elevation, mask=np.isnan(self.elevation))
        x_grid, y_grid = self.prep_grid()

        fig, ax = plt.subplots()
        cs = ax.contourf(y_grid, x_grid, z)
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('elevation [m]')
        ax.set_xlabel('Y [m]')
        ax.set_ylabel('X [m]')
        ax.axis('scaled')
        plt.gca().invert_yaxis()  # because top-left is our origin
        plt.show()


