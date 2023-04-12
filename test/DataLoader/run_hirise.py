import matplotlib.pyplot as plt

import pydem.DataLoader as dl

if __name__=="__main__":
    datadir = 'C:/Users/ktomita3/Documents/001_workspace/terrain4hda/data'
    imgname = 'DTEEC_041937_1920_042003_1920_L01.IMG'

    dtm = dl.HiriseDtm(datadir, imgname)
    dtm.visualize()