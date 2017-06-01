import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d


def printname(name):
    print name


def extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        paths_cache_folder=None):
    seg = vigra.readHDF5(seg_path, key)
    Volume = deepcopy(seg)
    all_paths = []
    paths_to_objs = []

    for i in np.unique(seg):
        if i == 0:
            continue

        # masking volume
        Volume[seg != i] = 0
        Volume[seg == i] = 1

        # skeletonize
        skel_img = skeletonize_3d(Volume)

        # conversion to array
        skel = np.array(
            [[e1, e2, e3] for e1, e2, e3 in zip(np.where(skel_img)[0], np.where(skel_img)[1], np.where(skel_img)[2])])

        # FIXME what happens if the distance transform cuts too much off so there is no more skeleton ?
        # if skel.shape[0] == 0:
        #     continue

        all_paths.extend([skel])
        paths_to_objs.extend([i])

    paths_to_objs = np.array(paths_to_objs, dtype="float64")
    all_paths = np.array(all_paths)

    return all_paths, paths_to_objs


def extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder=None):
    Volume = deepcopy(seg)
    all_paths = []
    paths_to_objs = []
    path_classes = []

    # no skeletons too close to the borders No.1
    dt = ds.inp(2)

    dt[dt < 12] = 0  # "sensitivity"

    for i in np.unique(seg):
        if i == 0:
            continue

        # masking volume
        Volume[seg != i] = 0
        Volume[seg == i] = 1

        # skeletonize
        skel_img = skeletonize_3d(Volume)

        # no skeletons too close to the borders No.2
        skel_img[dt == 0] = 0

        # conversion to array
        skel = np.array(
            [[e1, e2, e3] for e1, e2, e3 in zip(np.where(skel_img)[0], np.where(skel_img)[1], np.where(skel_img)[2])])

        # FIXME what happens if the distance transform cuts too much off so there is no more skeleton ?
        # if skel.shape[0] == 0:
        #     continue

        if len(np.unique(gt[skel_img == 1])) == 1:
            path_classes.extend([True])
        else:
            path_classes.extend([False])

        all_paths.extend([skel])
        paths_to_objs.extend([i])

    path_classes = np.array(path_classes)
    paths_to_objs = np.array(paths_to_objs, dtype="float64")
    all_paths = np.array(all_paths)

    return all_paths, paths_to_objs, path_classes, correspondence_list


if __name__ == "__main__":
    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_from_segmentation/input_ds.pkl', mode='r') as f:
        ds = pickle.load(f)
    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_from_segmentation/input_seg_path.pkl',
              mode='r') as f:
        seg_path = pickle.load(f)
    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_from_segmentation/input_key.pkl', mode='r') as f:
        key = pickle.load(f)
    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_from_segmentation/input_paths_cache_folder.pkl',
              mode='r') as f:
        paths_cache_folder = pickle.load(f)
    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_from_segmentation/output_all_paths.pkl',
              mode='r') as f:
        all_paths = pickle.load(f)
    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_from_segmentation/output_paths_to_objs.pkl',
              mode='r') as f:
        paths_to_objs = pickle.load(f)

    extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        paths_cache_folder=None)

    with open(
            '/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/input_correspondence_list.pkl',
            mode='r') as f:
        input_correspondence_list = pickle.load(f)

    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/input_ds.pkl',
              mode='r') as f:
        ds = pickle.load(f)

    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/input_gt.pkl',
              mode='r') as f:
        gt = pickle.load(f)

    with open(
            '/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/input_paths_cache_folder.pkl',
            mode='r') as f:
        paths_cache_folder = pickle.load(f)

    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/input_seg.pkl',
              mode='r') as f:
        seg = pickle.load(f)

    with open('/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/input_seg_id.pkl',
              mode='r') as f:
        seg_id = pickle.load(f)

    with open(
            '/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/output_all_paths.pkl',
            mode='r') as f:
        all_paths = pickle.load(f)

    with open(
            '/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/output_correspondence_list.pkl',
            mode='r') as f:
        output_correspondence_list = pickle.load(f)

    with open(
            '/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/output_path_classes.pkl',
            mode='r') as f:
        path_classes = pickle.load(f)

    with open(
            '/mnt/localdata03/amatskev/neuraldata/test/extract_paths_and_labels_from_segmentation/output_paths_to_objs.pkl',
            mode='r') as f:
        paths_to_objs = pickle.load(f)

    extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        paths_cache_folder=None)

    extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        seg_id,
        gt,
        input_correspondence_list,
        paths_cache_folder=None)
