import sys
sys.path.append(
    '/media/axeleik/EA62ECB562EC8821/data/pipeline/nature_methods_multicut_pipeline/software/')

print '####################################################################'
print 'Initializing ...'
print '####################################################################'
execfile('init_exp_splB_z1.py')

print '####################################################################'
print 'Running multicut ...'
print '####################################################################'
execfile('run_mc_splB_z1.py')

print '####################################################################'
print 'Detecting merges ...'
print '####################################################################'
execfile('detect_merges_splB_z1.py')

print '####################################################################'
print 'Resolving segmentation ...'
print '####################################################################'
execfile('resolve_splB_z1.py')
