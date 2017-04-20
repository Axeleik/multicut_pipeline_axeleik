
import sys

sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import MetaSet
from multicut_src import DataSet
from multicut_src import lifted_multicut_workflow
from multicut_src import ExperimentSettings
from multicut_src import merge_small_segments

import os
import vigra

cache_folder = '/media/axeleik/EA62ECB562EC8821/data/pipeline/erstellt/result_folder/cache/'
meta = MetaSet(cache_folder)

meta.load()

ds_train = meta.get_dataset('splB_z0')
ds_test = meta.get_dataset('splB_z1')

# id of the segmentation (== superpixels) we will calculate everything for
# we have added only one segmentation, so we have to use 0 here
seg_id = 0

# ExperimentSettings holds all relveant options for the experiments
# they are initialised to sensible defaults and
# we only have to change a few
mc_params = ExperimentSettings()

# cache folder for the RF
mc_params.set_rfcache(os.path.join(cache_folder, "rf_cache"))
# train RF with 500 trees
mc_params.set_ntrees(500)
# degree of anisotropy for the filter calculation
# (values bigger than 20 lead to calculation in 2d)
# set to 1. for isotropic data (default value)
anisotropy = 10.
mc_params.set_anisotropy(anisotropy)
# flag to indicate whether special z - edge features are computed
# set to false for isotropic data (default value)
mc_params.set_use2d(False)
# Threads
mc_params.set_nthreads(4)
# Solver
mc_params.set_solver("multicut_fusionmoves")
# mc_params.set_verbose(True)
mc_params.set_weighting_scheme("z")
# # For a lifted neighborhood
mc_params.set_lifted_neighborhood(3)

# list of features taken into account
# "raw" -> filters on raw data accumulated over the edges
# "prob" -> filters on probability maps accumulated over the edges
# "reg" -> region statistics, mapped to the edges
# "topo" -> topology features for the edges
feat_list = ("raw", "prob", "reg", "topo")
lifted_feat_list = ['reg', 'cluster']

# Call make filters
ds_train.make_filters(0, anisotropy)
ds_train.make_filters(1, anisotropy)

mc_nodes, mc_edges, mc_energy, t_inf = lifted_multicut_workflow(
    ds_train, ds_test,
    seg_id, seg_id,
    feat_list, lifted_feat_list,
    mc_params
)

# mc_nodes = result for the segments
# mc_edges = result for the edges
# mc_energy = energy of the solution
# t_inf = time the inference of the mc took
mc_nodes_train, mc_edges_train, mc_energy_train, t_inf_train = lifted_multicut_workflow(
    ds_train, ds_train,
    seg_id, seg_id,
    feat_list, lifted_feat_list,
    mc_params
)

# project the result back to the volume
mc_seg = ds_test.project_mc_result(seg_id, mc_nodes)
mc_seg_train = ds_train.project_mc_result(seg_id, mc_nodes_train)

# Write pre-result in case merge small segments errors out
vigra.writeHDF5(mc_seg, cache_folder + '../pre_result.h5', 'z/1/test')
vigra.writeHDF5(mc_seg_train, cache_folder + '../pre_result.h5', 'z/0/train')

# # Load pre-result
# mc_seg = vigra.readHDF5(cache_folder + '../pre_result.h5', 'z/1/test')
# mc_seg_train = vigra.readHDF5(cache_folder + '../pre_result.h5', 'z/0/train')

mc_seg, _, _ = vigra.analysis.relabelConsecutive(mc_seg, start_label=1, keep_zeros=False)
mc_seg_train, _, _ = vigra.analysis.relabelConsecutive(mc_seg_train, start_label=1, keep_zeros=False)

# merge small segments
mc_seg = merge_small_segments(mc_seg, 100)
mc_seg_train = merge_small_segments(mc_seg_train, 100)

vigra.writeHDF5(mc_seg, cache_folder + '../result.h5', 'z/1/test')
vigra.writeHDF5(mc_seg_train, cache_folder + '../result.h5', 'z/0/train')


