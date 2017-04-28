import h5py
import numpy as np
import vigra
import cPickle as pickle
from scipy import interpolate


def printname(name):
    print name

if __name__ == "__main__":

    f = h5py.File("/media/axeleik/EA62ECB562EC8821/data/pipeline/erstellt/cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5", mode="r")

    f.visit(printname)
    print np.array(f["z/1/labels"])