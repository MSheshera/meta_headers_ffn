"""
Mainly contains global variables for feature extraction. Will also allow for 
changing of these variables which then get imported everywhere.
"""
# Settings to use to create the simstring databases to do look ups from at 
# time of feature extraction.

## Path to the directory which contains the text lexicon files. 
lexicon_dir = '/home/msheshera/MSS/Code/Projects/meta_headers/keyword_lexicons/dicts'
## The destination directory where you want the simstring databases to be 
## written.
simstringdb_dir = '/home/msheshera/MSS/Code/Projects/meta_headers/sm-deep-metadata-extraction/feed_fwd_nw/simstringdb'
## Names of the files in dict_dir. These must match.
people = ['people_frequent_last_names.txt', 'people_shared.txt',
		'people_english_only.txt', 'people_chinese_only.txt']
places = ['city_full.txt', 'country_full.txt', 'region_full.txt']
departments = ['department_full.txt', 'department_keywords.txt', 
'faculty_full.txt', 'faculty_keywords.txt']
universities = ['institution_full.txt', 'institution_keywords.txt', 'university_full.txt', 'university_keywords.txt']            


# Settings to use to extract features.
## Top level GROTOAP directory. Both grotoap datasets below should contain the
## numbered directories with the .cxml files in them.
## For example: grotoap_dir_train/00/*.cxml grotoap_dir_train/01/*.cxml etc.
grotoap_dir_train = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/train_dataset'
grotoap_dir_test = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/test_dataset'
## Directory to write extracted feature matrices (npy files) into.
out_dir = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/npy_out'
## Directories to which maps (what int value represents what label etc) should 
## be written.
map_dir = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/map_dir'
## Directory with the simstring databases. The directory with the databases
## which was created above.
simstringdb_dir = simstringdb_dir

## How the x and y coordinates of the word should be binned.
num_x_bins = 4
num_y_bins = 4
use_lexicons = True
## This hasnt been tested of fully implemented. Set to False always.
debug = False