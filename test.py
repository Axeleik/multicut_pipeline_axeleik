import h5py
import numpy as np
import vigra
import cPickle as pickle
from scipy import interpolate

"""
with open("/media/axeleik/EA62ECB562EC8821/data/test/concatenated.pkl", 'r') as f:
    concan= pickle.load(f)
print concan




for number,data in enumerate(paths):
    print data,",",number


    data = np.array([(elem1, elem2, elem3*10) for elem1, elem2, elem3 in data])
    data = data.transpose()

    tck, u = interpolate.splprep(data, s=3500, k=3)

    new = interpolate.splev(np.linspace(0, 1, 100000), tck)

    data=np.array(new).transpose()
    paths[number]=data
    

with open("/media/axeleik/EA62ECB562EC8821/data/test/stetig.pkl", 'w') as f:
    pickle.dump(paths, f)

with open("/media/axeleik/EA62ECB562EC8821/data/test/stetig.pkl", 'r') as f:
    paths2= pickle.load(f)

print len(paths1)
print paths1[0].shape
print len(paths2)
print paths2[0].shape
"""

execfile("run_mc.py")
execfile("false_merges_2.py")