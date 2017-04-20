import os
import vigra
import numpy as np
import cPickle as pickle

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

# hack to get in meta
# import sys
# sys.path.append('..')
# from init_exp import meta
from multicut_src import MetaSet
from multicut_src import DataSet

# from multicut_src import shortest_paths, path_feature_aggregator
from multicut_src import compute_false_merges, resolve_merges_with_lifted_edges
from multicut_src import project_resolved_objects_to_segmentation
from multicut_src import ExperimentSettings

results_folder = '/media/axeleik/EA62ECB562EC8821/data/pipeline/erstellt/result_folder/'
cache_folder = "/media/axeleik/EA62ECB562EC8821/data/pipeline/erstellt/result_folder/cache/"
source_folder = '/media/axeleik/EA62ECB562EC8821/data/pipeline/erstellt/'


def find_false_merges(ds_str):

    meta = MetaSet(cache_folder)
    meta.load()
    ds = meta.get_dataset(ds_str)

    # Load train datasets: for each source
    train_raw_sources = [
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_raw_sources_keys = [
        'z/0/raw',
        'z/1/raw',
        'z/0/raw',
        'z/0/raw',
        'z/1/raw'
    ]
    train_probs_sources = [
        source_folder + 'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_probs_sources_keys = [
        'z/0/data',
        'z/1/data',
        'z/0/data',
        'z/0/data',
        'z/1/data'
    ]
    gtruths_paths = [
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    gtruths_keys = [
        'z/0/neuron_ids',
        'z/1/neuron_ids',
        'z/0/neuron_ids',
        'z/0/neuron_ids',
        'z/1/neuron_ids'
    ]
    ds_names = [
        'splA_z0',
        'splA_z1',
        'splB_z0',
        'splC_z0',
        'splC_z1'
    ]
    trainsets = []
    for id_source, raw_source in enumerate(train_raw_sources):
        trainsets.append(
            DataSet(
                cache_folder, 'ds_train_{}'.format(ds_names[id_source])
            )
        )
        trainsets[-1].add_raw(raw_source, train_raw_sources_keys[id_source])
        trainsets[-1].add_input(train_probs_sources[id_source], train_probs_sources_keys[id_source])
        trainsets[-1].add_gt(gtruths_paths[id_source], gtruths_keys[id_source])

    train_segs = [
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9
    ]
    test_seg = cache_folder + '../result.h5'

    train_keys = [
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7'],
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7']
    ]
    test_key = 'z/1/test'
    rf_save_folder = cache_folder + 'rf_cache/path_rfs'

    paths_save_folder = cache_folder + 'path_data/'
    train_paths_load_folder = results_folder + 'paths_cache/train/'

    params = ExperimentSettings()
    params.set_anisotropy(10.)

    paths, false_merge_probs, path_to_objs = compute_false_merges(
        trainsets,
        ds,
        train_segs,
        train_keys,
        test_seg,
        test_key,
        rf_save_folder,
        paths_save_folder=paths_save_folder,
        train_paths_save_folder=train_paths_load_folder,
        params=params
    )
    print "Succesfully computed false merge probabilities for", len(false_merge_probs), "paths"
    # import os
    # os.mkdir('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/path_data/')
    # with open('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/path_data/paths.pkl', 'w') as f:
    #     pickle.dump(paths, f)
    with open(cache_folder + 'path_data/false_paths_predictions.pkl', 'w') as f:
        pickle.dump(false_merge_probs, f)
    # with open('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/path_data/paths_to_objs.pkl', 'w') as f:
    #     pickle.dump(path_to_objs, f)


def resolve_false_merges_threshold_test_settings(mc_params, meta):

    test_set_name = 'splB_z1'
    weight_file_name = 'probs_to_energies_0_3926342027172393727_rawprobregtopo.h5'

    from resolve_false_merges import resolve_merges_threshold_test
    resolve_merges_threshold_test(
        meta, test_set_name,
        mc_params, cache_folder,
        weight_file_name,
        min_prob_thresh=0.5
    )


if __name__ == '__main__':

    # 1.) find false merge objects
    find_false_merges('splB_z1')

    # # 2.) resolve the objs classified as false merges
    # # parameters for the Multicut
    # meta = MetaSet(cache_folder)
    # mc_params = ExperimentSettings()
    # rfcache = os.path.join(meta.meta_folder, "rf_cache")
    # mc_params.set_rfcache(rfcache)
    #
    # mc_params.set_anisotropy(10.)
    # mc_params.set_use2d(False)
    #
    # mc_params.set_nthreads(30)
    #
    # mc_params.set_ntrees(500)
    #
    # mc_params.set_solver("multicut_fusionmoves")
    # # mc_params.set_verbose(True)
    # mc_params.set_weighting_scheme("z")
    #
    # mc_params.set_lifted_neighborhood(3)
    #
    # mc_params.min_nh_range = 5
    # mc_params.max_sample_size = 20
    # # mc_params.max_sample_size = 10
    #
    # # resolve_false_merges(mc_params)
    # resolve_false_merges_threshold_test_settings(mc_params, meta)

    # # 3.) project the resolved result to segmentation
    # project_new_segmentation()