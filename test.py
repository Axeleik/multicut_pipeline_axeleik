import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d
from Queue import LifoQueue
from time import time
import h5py
import nifty_with_cplex as nifty
import matplotlib.pyplot as plt



def cut_off(all_paths_unfinished,paths_to_objs_unfinished,cut_off_array,ratio_true=0.13,ratio_false=0.4):
    print "start cutting off array..."
    test_label = []
    con_label = {}
    test_length = []
    con_len = {}

    for label in cut_off_array.keys():
        con_label[label]=[]
        con_len[label]=[]
        for path in cut_off_array[label]:
            test_label.append(path[0])
            con_label[label].append(path[0])
            test_length.append(path[1])
            con_len[label].append(path[1])

    help_array=[]
    for label in con_label.keys():
        conc=np.concatenate(con_label[label]).tolist()
        counter=[0,0]
        for number in np.unique(conc):
            many = conc.count(number)

            if counter[1] < many:
                counter[0] = number
                counter[1] = many

        for i in xrange(0,len(con_label[label])):
            help_array.extend([counter[0]])





    end = []

    for idx, path in enumerate(test_label):


        overall_length = 0
        for i in test_length[idx]:
            overall_length = overall_length + i

        less_length = 0
        for u in np.where(np.array(path) != help_array[idx]):
            for index in u:
                less_length = less_length + test_length[idx][index]

        end.extend([less_length/ overall_length])



    path_classes=[]
    all_paths=[]
    paths_to_objs=[]
    for idx,ratio in enumerate(end):
        if ratio<ratio_true:
            path_classes.extend([True])
            all_paths.extend([all_paths_unfinished[idx]])
            paths_to_objs.extend([paths_to_objs_unfinished[idx]])

        elif ratio>ratio_false:
            path_classes.extend([False])
            all_paths.extend([all_paths_unfinished[idx]])
            paths_to_objs.extend([paths_to_objs_unfinished[idx]])





    print "finished cutting of"
    return np.array(all_paths),np.array(paths_to_objs, dtype="float64"),np.array(path_classes)


def bar_plot():
    print "loading array..."
    end = np.load("/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/to_plot.npy")
    print "array loaded"


    y = np.sort(end)[np.where(np.sort(end) == 0)[0][-1] + 1:]
    N = len(y)
    x = range(N)

    plt.bar(x, y, color="blue")

    fig = plt.gcf()
    plt.show()




if __name__ == "__main__":
    pass







