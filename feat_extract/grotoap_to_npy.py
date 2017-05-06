"""
Generate npy files for the feed forward neural network.
Each word is an example; each word has the following features:
shape, word length, width (spatial), height (spatial), wh_ratios, 
x_coords, y_coords, page_ids, line_ids, zone_ids, place_score, 
department_score, university_score, person_score.
"""
import os, sys, copy, errno, time
import codecs, pprint, collections, pickle
import numpy as np
import simstring, re
from scipy import spatial
# Make sure you can print utf-8 to the console if need be.
sys.stdout=codecs.getwriter('utf-8')(sys.stdout)

# XML parsing code.
import parse_docs_sax as mpds

# Type of dataset you're creating.
#dataset_str = 'test'
# Top level GROTOAP directory.
grotoap_dir_train = '/mnt/6EE804CEE804970D/Academics/Projects/meta_headers/Dataset/project_subset/train'
grotoap_dir_test = '/mnt/6EE804CEE804970D/Academics/Projects/meta_headers/Dataset/project_subset/test'
# Directory to write TFRecord files into.
out_dir = '/mnt/6EE804CEE804970D/Academics/Projects/meta_headers/Dataset/npy_out'
# Directories to which maps should be written.
map_dir = './map_dir'
# Directory with the simstring databases
simstringdb_dir = './simstringdb'
# How the x and y coordinates of the word should be binned.
num_x_bins = 4
num_y_bins = 4
use_lexicons = True
debug = False
# Create a dataframe to which everything gets appended and then written to disk.
features_list = list()
# TODO: Update all comments to say that you're doing numpy and not TFRecords.

##########################################################
#      Functions to help with feature extraction.        #
##########################################################

def get_neigh_words(prev_line_g, pres_line_g, next_line_g):
    """
    Given the previous, next and present line of words; returns the 
    neighbour words of all words in the present line.
    """
    # Copy lists over so they're not changed :/
    pres_len = len(pres_line_g.words)
    prev_line = copy.deepcopy(prev_line_g)
    pres_line = copy.deepcopy(pres_line_g)
    next_line = copy.deepcopy(next_line_g)
    
    # First pad line with end and beginning words.
    for line in [prev_line, pres_line, next_line]:
        line.words.append(line.words[-1])
        line.words.insert(0,line.words[0])
    # Find the centroids of all the words.
    top_coords = np.array([w.centerpoint() for w in prev_line.words])
    mid_coords = np.array([w.centerpoint() for w in pres_line.words])
    bot_coords = np.array([w.centerpoint() for w in next_line.words])
    # Find eucledian distances to each of the top and bottom words.
    top_dist = spatial.distance.cdist(mid_coords, top_coords)
    bot_dist = spatial.distance.cdist(mid_coords, bot_coords)
    # For the words in the middle find the top and bottom neighbours. 
    allword_neigh = list()
    for w_i in range(1, pres_len+1):
        word_neigh = list()
        # Adding 1 because of padding.
        topn_i = np.argmin(top_dist[w_i,1:-1])+1
        botn_i = np.argmin(bot_dist[w_i,1:-1])+1
        # Add neighbouring words.
        word_neigh.extend([prev_line.words[topn_i-1],
            prev_line.words[topn_i], prev_line.words[topn_i+1]])
        word_neigh.extend([pres_line.words[w_i-1], pres_line.words[w_i+1]])
        word_neigh.extend([next_line.words[botn_i-1],
            next_line.words[botn_i], next_line.words[botn_i+1]])
        allword_neigh.append(word_neigh)
    # for word in allword_neigh:
    #     print [w.text for w in word]
    return allword_neigh

def score_string(word_text, dbname):
    """
    Impliments the simstring matching. 
    Assumes the presence of the simstring databases.
    """
    db = simstring.reader(os.path.join(simstringdb_dir, dbname))
    db.measure = simstring.cosine
    db.threshold = 0.6
    if db.retrieve(word_text.encode('utf-8')):
        return 1
    else:
        return 0

def check_numeric_type(word_text):
    """
    Given word text says if string is a number, alpha-numeric or neither.
    http://stackoverflow.com/a/20929881/3262406
    """
    try:
        # token is a number
        float(word_text)
        return 1
    except ValueError:
        # token is alpha-numeric
        if word_text.isalnum():
            return 2
    # token is neither (so just text)
    return 0

def check_email(word_text):
    if re.match(r"[^@]+@[^@]+\.[^@]+", word_text):
        return 1
    return 0

##########################################################
#           Functions to serialize data.                 #
##########################################################

def serialize_example(intmapped_label, 
                    intmapped_shapes, word_lens, word_num_type, word_email,
                    widths, heights, wh_ratios, x_coords, y_coords,
                    zones, lines,
                    place_scores, department_scores, university_scores,
                    person_scores):
    """
    Given a writer and the features for an example serialzie it and write it 
    to the TFRecord file.
    """
    global features_list
    row = np.concatenate((
        intmapped_shapes, # One hot. 4 values (0,1,2,3). 9 values.
        word_lens, # Can normalize. 9 values.
        word_num_type, # One hot. 3 values (0,1,2). 9 values.
        word_email, # Binary. 9 values.
        widths, # Normalize. 9 values.
        heights, # Normalize. 9 values.
        wh_ratios, # Normalize. 9 values.
        x_coords, # Can normalize. 9 values. Discrete ordinal.
        y_coords, # Can normalize. 9 values. Discrete ordinal.
        zones, # Normalize. 9 values.
        lines, # Normalize. 9 values.
        place_scores, # Binary. 9 values.
        department_scores, # Binary. 9 values.
        university_scores, # Binary. 9 values.
        person_scores, # Binary. 9 values.
        intmapped_label))

    features_list.append(row)

def process_tok(words_tl, label_map, shape_map,
                label_distr, max_x, max_y, min_x, min_y):
    """
    Do the actual feature extraction for a word. 
    words_tl: [(word instance,(page_id, zone_id, line_id)),...]
    """
    # last word in list of word tuples is target word
    tar_word, _ = words_tl[-1]
    # scalar label id
    intmapped_label = np.zeros(1, dtype=np.float64)
    # Create the following feature vectors for each word.
    # (Neighbour features are part of target word features, just saying.)
    # vector of shape ids
    intmapped_shapes = np.zeros(9, dtype=np.float64)
    word_lens = np.zeros(9, dtype=np.float64)
    # If the word is a number/alphanumeric/just text.
    word_num_type = np.zeros(9, dtype=np.float64)
    # If the word is an email address
    word_email = np.zeros(9, dtype=np.float64)
    # geometrical information:
    widths = np.zeros(9, dtype=np.float64)
    heights = np.zeros(9, dtype=np.float64)
    wh_ratios = np.zeros(9, dtype=np.float64)
    x_coords = np.zeros(9, dtype=np.float64)
    y_coords = np.zeros(9, dtype=np.float64)
    # enclosing region information (keep track of what zone, page, 
    # or line each word is in)
    # Skipping pages for now because they're not being used.
    #pages = np.zeros(9, dtype=np.float64)
    zones = np.zeros(9, dtype=np.float64)
    lines = np.zeros(9, dtype=np.float64)
    # lexicon matches
    place_scores = np.zeros(9, dtype=np.float64)
    department_scores = np.zeros(9, dtype=np.float64)
    university_scores = np.zeros(9, dtype=np.float64)
    person_scores = np.zeros(9, dtype=np.float64)

    # Get label and add to label_map
    label = tar_word.label
    if label not in label_map:
        print("process_tok: Label {:s} not in label map; Adding.".format(label))
        label_map[label] = len(label_map)+1
    intmapped_label[0] = label_map[label]
    # Form count to get class distribution.
    label_distr[label] += 1

    # Extract features for each word.    
    for i, word_t in enumerate(words_tl):
        word, (pid, zid, lid) = word_t
        
        # Get geometric features
        width = word.width()
        height = word.height()
        if height != 0:
            wh_ratio = width/height
        else:
            wh_ratio = 0
        (x_coord, y_coord) = word.centerpoint()
        
        # Update feature arrays
        # Get shape and add to shape map
        word_shape = word.shape()
        if word_shape not in shape_map:  # and update_vocab:
            shape_map[word_shape] = len(shape_map)+1
        intmapped_shapes[i] = shape_map[word_shape]

        # Word numeric type
        word_num_type[i] = check_numeric_type(word.text)
        # Check if word is an email address
        word_email[i] = check_email(word.text)

        word_lens[i] = len(word.text)
        # Geometric information:
        widths[i] = width
        heights[i] = height
        wh_ratios[i] = wh_ratio
        x_coords[i] = x_coord
        y_coords[i] = y_coord
        # Enclosing region information (keep track of what zone, page, 
        # or line each word is in)
        #pages[i] = pid
        zones[i] = zid
        lines[i] = lid
        # Lexicon matches
        if use_lexicons:
            place_scores[i] = score_string(word.text, dbname='places.db')
            department_scores[i] = score_string(word.text, dbname='departments.db')
            university_scores[i] = score_string(word.text, dbname='universities.db')
            person_scores[i] = score_string(word.text, dbname='people.db')

    # Bin the x and y coordinates (4 bins from 0 to max x, y on page)
    x_bins = np.linspace(min_x, max_x, num=num_x_bins)
    x_coords = np.digitize(x_coords, x_bins)
    y_bins = np.linspace(min_y, max_y, num=num_y_bins)
    y_coords = np.digitize(y_coords, y_bins)

    # Print debug information
    if debug:
        print('intmapped_shapes:',intmapped_shapes)
        print('word_lens:', word_lens)
        print('intmapped_label:', intmapped_label)
        # geometrical information:
        print('widths:', widths)
        print('heights:', heights)
        print('wh_ratios:', wh_ratios)
        print('x_coords:', x_coords)
        print('y_coords:', y_coords)
        # enclosing region information (keep track of what zone, page, 
        # or line each word is in)
        #print('pages:', pages)
        print('zones:', zones)
        print('lines:', lines)
        # lexicon matches
        if use_lexicons:
            print('place_scores:', place_scores)
            print('department_scores:', department_scores)
            print('university_scores:', university_scores)
            print('person_scores:', person_scores)

    # Serialize the example.
    serialize_example(intmapped_label,
                    intmapped_shapes,
                    word_lens,
                    word_num_type,
                    word_email,
                    widths,
                    heights,
                    wh_ratios,
                    x_coords,
                    y_coords,
                    zones,
                    lines,
                    place_scores,
                    department_scores,
                    university_scores,
                    person_scores)

def make_examples(page, label_map, shape_map, label_distr):
    """
    Given a document page make examples out of the words.
    """
    # Find the max and min x and y coordinates for binning.
    max_x = 0; max_y = 0
    min_x = 100000; min_y = 100000
    for zone in page.zones:
        for line in zone.lines:
            for word in line.words:
                (x,y) = word.centerpoint()
                if x > max_x:
                    max_x = x
                elif x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                elif y < min_y:
                    min_y = y
    
    # Get the 8 words surrounding the target word; these form part of the 
    # present words example.
    # First get all lines into a list to avoid crazy edge case code.
    # Pad with the first and last line with itself; like an image padded 
    # with itself at the edges.
    lines = list()
    lines.append(copy.deepcopy(page.zones[0].lines[0]))
    # Associate a page.id, zone.id and a line.id with each word. 
    # page and zone id needs to be captured here.
    struct_ids = list()
    struct_ids.append((page.id, page.zones[0].id, page.zones[0].lines[0].id))
    for zone in page.zones:
        for line in zone.lines:
            # If a line is empty ignore it. 
            # Might not be the best thing to do.
            if line.words:
                lines.append(line)
                struct_ids.append((page.id, zone.id, line.id))
    lines.append(copy.deepcopy(line))
    struct_ids.append((page.id, zone.id, line.id))

    # Now look three lines at a time and find words and the neighbours.
    word_count = 0 
    for li in range(1, len(lines)-1):
        prev_line = lines[li-1]
        pres_line = lines[li]
        next_line = lines[li+1]
        # Find neighbours of all words in present line. Same len as line.
        line_neigh_list = get_neigh_words(prev_line, pres_line, next_line)
        for word, w_neigh_list in zip(lines[li].words, line_neigh_list):
            words_tl = list()
            # Associate a page and zone id with neighbour words.
            neigh_struct_ids = [struct_ids[li-1],struct_ids[li-1],
                struct_ids[li-1], struct_ids[li],struct_ids[li],
                struct_ids[li+1],struct_ids[li+1],struct_ids[li+1]]
            words_tl = [tuple((w,si)) for w,si in
                zip(w_neigh_list,neigh_struct_ids)]
            # Add target word at end of list of neighbours.
            word_t = tuple((word, struct_ids[li]))
            words_tl.append(word_t)
            process_tok(words_tl, label_map, shape_map,
                label_distr, max_x, max_y, min_x, min_y)
            word_count += 1
    return word_count

def doc_to_example(in_docname, label_map, shape_map, label_distr):
    """
    Make a given file into a set of examples to be written to a TFRecord 
    file.
    Input:
        tfr_writer: The TFRecord writer.
        in_docname: The name of the .cxml file to process.
        out_dir: The output directory into which all TFRecord files
            are getting written to.
    """
    doc = mpds.parse_doc(in_docname)
    # Everything only for the first page.
    doc_word_count = make_examples(doc.pages[0], label_map, shape_map, label_distr)
    print('doc_to_example: Done processing {:d} words from: {:s}'.format(doc_word_count, os.path.basename(in_docname)))
    
    return doc_word_count

def dir_to_examples(in_dir, label_map, shape_map, label_distr):
    """
    Make entire directory of TrueViz files into one tensorflow record.
    """
    dir_file_count = 0
    dir_doc_word_count = 0
    # Go over each file in numbered sub directory and make it the
    # TFRecord file.
    print('dir_to_examples: Converting files in directory {:s}'
        .format(os.path.basename(in_dir)))
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if '.cxml' in file:
                in_docname = os.path.join(root, file)
                dir_doc_word_count += doc_to_example(in_docname,
                    label_map, shape_map, label_distr)
                dir_file_count += 1
    
    print('dir_to_examples: Done processing directory {:s} with {:d} files with {:f} words (tokens) on average'
        .format(os.path.basename(in_dir), dir_file_count, dir_doc_word_count/float(dir_file_count)))
    return dir_file_count, dir_doc_word_count

def grotoap_to_examples(grotoap_dir, out_dir, label_map, shape_map, label_distr):
    """
    Make entire directory (containing sub directories) of grotoap data to 
    TFRecords; one TFRecord file per sub-directory.
    Input:
        grotoap_dir: The top level directory of the dataset containing
            the 00, 01 etc named subdirectories.
        out_dir: The directory into which the TFRecord files should be
            written.
    """
    # Create output direcory if it doesnt exist.
    try:
        os.makedirs(out_dir)
        print('grotoap_to_examples: Created {} for TFRecord files'.format(out_dir))
    except OSError as ose:
        # For the case of *file* by name of out_dir existing
        if (not os.path.isdir(out_dir)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create output directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.
                errno, ose.strerror, out_dir))
            sys.exit(1)
    
    # Create list of output file names and list of directory basenames to 
    # read from.
    in_subdirs = list()
    for root, subdirs, files in os.walk(grotoap_dir):
        for subdir in subdirs:
            in_subdirs.append(subdir)
    
    # Create output filenames.
    in_subdirs.sort()
    out_fname = in_subdirs[0]+'-'+in_subdirs[-1]
    
    # Call dir_to_example for each directory in the grotoap directory
    file_count = 0
    total_word_count = 0
    for subdir in in_subdirs:
        in_dir = os.path.join(grotoap_dir, subdir)
        dir_file_count, dir_doc_word_count = dir_to_examples(in_dir, label_map, 
            shape_map, label_distr)
        file_count += dir_file_count
        total_word_count += dir_doc_word_count
    
    print('grotoap_to_examples: Total number of words (tokens) is: {:d}'
        .format(total_word_count))
    print('grotoap_to_examples: Processed {:d} files from {:d} directories'
        .format(file_count, len(in_subdirs)))
    return out_fname

def main(dataset_str):
    """
    Calls everything that needs to be from above.
    """
    # Check for the existance of the simstring databases.
    start_time = time.time()
    if os.path.isdir(simstringdb_dir):
        dep = os.path.isfile(os.path.join(simstringdb_dir, 'departments.db'))
        pla = os.path.isfile(os.path.join(simstringdb_dir, 'places.db'))
        peo = os.path.isfile(os.path.join(simstringdb_dir, 'people.db'))
        uni = os.path.isfile(os.path.join(simstringdb_dir, 'universities.db'))
        if not (dep and pla and peo and uni):
            sys.stderr.write('ERROR: Make sure simstring databases exist\n')
            sys.exit(-1)
    else:
        sys.stderr.write('ERROR: Make sure simstring directory exists\n')
        sys.exit(-1)

    # Create label and shape maps and call the whole thing.
    # If creating the train set, build the maps as you go.
    if dataset_str is 'train':
        print('main: Generating serialized {:s} set'.format(dataset_str))
        label_map = dict()
        shape_map = dict()
        label_distr = collections.defaultdict(int)
        out_fname = grotoap_to_examples(grotoap_dir_train, out_dir, label_map, shape_map, label_distr)        
        # Global variable features list has been updated. Now write it to disk.
        features_np = np.array(features_list)
        out_fname = os.path.join(out_dir, out_fname+'_'+dataset_str+'.npy')
        np.save(file=out_fname, arr=features_np)
        print('main: Wrote {:s}; shape:{}'.format(out_fname, features_np.shape))


        # Print out (inverted) maps for viewing pleasure :-P
        print('main: Label Map:')
        pprint.pprint({v: k for k, v in label_map.items()})
        print('\n')

        print('main: Shape Map:')
        pprint.pprint({v: k for k, v in shape_map.items()})
        print('\n')

        print('main: Label distribution:')
        label_sum = sum(label_distr.values())
        for k in label_distr.keys():
            label_distr[k] = label_distr[k]/float(label_sum)
        pprint.pprint(dict(label_distr))
        print('\n')
        
        # Save maps to disk
        save_maps(map_dir, label_map, shape_map, label_distr, dataset_str+'-skl')
    
    # If creating the test set use the existing maps.
    elif dataset_str in ['test', 'dev']:
        print('main: Generating serialized {:s} set'.format(dataset_str))

        label_map_path = os.path.join(map_dir,'label_map-train-skl.pd')
        shape_map_path = os.path.join(map_dir,'shape_map-train-skl.pd')
        label_distr = collections.defaultdict(int)
        
        # Check for existance of maps
        if os.path.isdir(map_dir):
            label = os.path.isfile(label_map_path)
            shape = os.path.isfile(shape_map_path)
            if not (label and shape):
                sys.stderr.write('ERROR: Make sure maps from training set exist\n')
                sys.exit(-1)
        else:
            sys.stderr.write('ERROR: Make sure maps directory exists\n')
            sys.exit(-1)
        
        # If it exists then read them in and pass them through.
        with open(label_map_path, 'rb') as pf:
            label_map = pickle.load(pf)
            print('main: Using existing {:s} label map'.format(label_map_path))
        with open(shape_map_path, 'rb') as pf:
            shape_map = pickle.load(pf)
            print('main: Using existing {:s} shape map'.format(shape_map_path))
        out_fname = grotoap_to_examples(grotoap_dir_test, out_dir, label_map, shape_map, label_distr)
        # Global variable features list has been updated. Now write it to disk.
        features_np = np.array(features_list)
        out_fname = os.path.join(out_dir, out_fname+'_'+dataset_str+'.npy')
        np.save(file=out_fname, arr=features_np)
        print('main: Wrote {:s}; shape:{}'.format(out_fname, features_np.shape))

        # Print out (inverted) maps for viewing pleasure :-P
        print('main: Label Map:')
        pprint.pprint({v: k for k, v in label_map.items()})
        print('\n')

        print('main: Shape Map:')
        pprint.pprint({v: k for k, v in shape_map.items()})
        print('\n')

        print('main: Label distribution:')
        label_sum = sum(label_distr.values())
        for k in label_distr.keys():
            label_distr[k] = label_distr[k]/float(label_sum)
        pprint.pprint(dict(label_distr))
        print('\n')


        # Save maps to disk
        save_maps(map_dir, label_map, shape_map, label_distr, dataset_str+'-skl')
    end_time = time.time()
    print('main: Time taken: {:f}s'.format(end_time-start_time))


##########################################################
#          Just some code to check correctness etc       #
##########################################################

def save_maps(map_dir, label_map, shape_map, label_distr, suffix_str):
    """
    Save dictionaries to pickle dumps.
    """
    # Create directory for map pickle dictionaries.
    try:
        os.makedirs(map_dir)
        print 'save_maps: Created {} for map files'.format(map_dir)
    except OSError as ose:
        # For the case of *file* by name of out_dir existing
        if (not os.path.isdir(out_dir)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create map directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.
                errno, ose.strerror, out_dir))
            sys.exit(1)
    label_path = os.path.join(map_dir, 'label_map-'+suffix_str+'.pd')
    with open(label_path,'wb') as lmf:
        pickle.dump(label_map, lmf)
    print('save_maps: Wrote: {:s}'.format(label_path))

    shape_path = os.path.join(map_dir, 'shape_map-'+suffix_str+'.pd')
    with open(shape_path,'wb') as smf:
        pickle.dump(shape_map, smf)
    print('save_maps: Wrote: {:s}'.format(shape_path))

    labeld_path = os.path.join(map_dir, 'label_distr-'+suffix_str+'.pd')
    with open(labeld_path,'wb') as ldf:
        pickle.dump(label_distr, ldf)
    print('save_maps: Wrote: {:s}'.format(labeld_path))

if __name__ == '__main__':
    name = 'train'
    main(name)