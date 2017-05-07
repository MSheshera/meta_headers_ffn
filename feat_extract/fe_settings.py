"""
Mainly contains global variables for feature extraction. Will also allow for 
changing of these variables which then get imported everywhere.
"""

# Top level GROTOAP directory.
grotoap_dir_train = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/train'
grotoap_dir_test = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/test'
# Directory to write npy files into.
out_dir = '/mnt/nfs/work1/mccallum/ashastry/smysore-data/npy_out'
# Directories to which maps should be written.
map_dir = '/mnt/nfs/work1/mccallum/ashastry/smysore-data/map_dir'
# Directory with the simstring databases
simstringdb_dir = '/mnt/nfs/work1/mccallum/smysore/grotoap-dataset/project_subset/lex_keywords_db'

# How the x and y coordinates of the word should be binned.
num_x_bins = 4
num_y_bins = 4
use_lexicons = True
debug = False