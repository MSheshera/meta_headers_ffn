"""
Mainly contains global variables and such. Will also allow for 
changing of these variables which then get imported everywhere.
"""
import os
# LOCAL paths.
# Dataset paths.
l_train_path = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/npy_out/00-02_train.npy'
l_dev_path = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/npy_out/03-03_test.npy'
l_test_path = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/npy_out/03-03_test.npy'
# Directories from which to read maps for the labels.
l_map_path = '/home/msheshera/MSS/Code/Projects/meta_headers/sm-deep-metadata-extraction/feed_fwd_nw/map_dir/'
l_map_path = os.path.join(l_map_path, 'label_map-train-skl.pd')
# Chcekpoint directory. Just saving models here now.
l_check_dir = '/home/msheshera/MSS/Code/Projects/meta_headers/sm-deep-metadata-extraction/feed_fwd_nw/check_dir'

# SWARM2 paths.
s_train_path = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/npy_out/00-05_train.npy'
s_dev_path = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/npy_out/98-99_dev.npy'
s_test_path = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/npy_out/21-77_test.npy'
# Read maps for the labels.
s_map_path = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/map_dir/'
s_map_path = os.path.join(s_map_path, 'label_map-train-skl.pd')
# Chcekpoint directory. Just saving models here now.
s_check_dir = '/home/smysore/meta_headers/feed_fwd_nw/check_dir'

# Desired labels.
desired_labels = (u'AUTHOR_TITLE', u'AFFILIATION', u'ABSTRACT', u'AUTHOR', u'TITLE')

# Feature types.
# Manually specifying these annoying masks now. But should consider using
# a pandas dataframe with named columns and just calling things by name.
binary_mask = [False]*(3*9)+[True]*9+[False]*(7*9)+[True]*(4*9)
norm_mask = [False]*9+[True]*9+[False]*(2*9)+[True]*(7*9)+[False]*(4*9)
categorical_mask = [True]*9+[False]*9+[True]*9+[False]*(9*12)
