"""
File contains utility code to just read data in and such.
"""
import os, sys
import pickle
import numpy as np
from sklearn.externals import joblib

def read_data(data_path, map_path, desired_labels):
    """
    Read the data in from the numpy binary format and set undesired labels
    to 'OTHER'.
    Input:
        data_path: Path to the numpy file to load.
        map_path: A path to the saved string to int_label map; saved as a 
            python dictionary.
        desired_labels: A list of strings which are the labels we want to
            work with.   
    """
    # Read data in.
    dataset = np.load(data_path)
    
    # Read label maps in.
    with open(map_path, 'rb') as pf:
        label_map = pickle.load(pf)
    
    # Build a map of desired labels.    
    desired_map = dict([(l_str,label_map[l_str]) for 
        l_str in desired_labels])
    mask = np.array([False]*dataset.shape[0])
    # Build a mask with all labels not in int_labels are set to False.
    for int_label in desired_map.values():
        temp = dataset[:,-1]==int_label
        mask = np.logical_or(mask, temp)
    # Invert the mask so you can index into the dataset.
    mask = np.logical_not(mask)
    # Add OTHER label to map
    desired_map[u'OTHER'] = 23
    # Set all labels not in int_labels to 'OTHER'.
    dataset[mask,-1] = desired_map[u'OTHER']

    dataset_x = dataset[:,:-1]
    dataset_y = dataset[:,-1]
    print('Read data: {}'.format(data_path))
    return dataset_x, dataset_y, desired_map

def load_saved_model(model_path):
    """
    Takes a path to a valid saved model and returns it.
    """
    model = joblib.load(model_path)
    print('Read model from: {:s}'.format(model_path))
    sys.stdout.flush()
    return model

def save_model(model, model_path):
    """
    Takes a learnt model and saves it to the model_path.
    """
    joblib.dump(model, model_path)
    print('Wrote model to: {:s}'.format(model_path))
    sys.stdout.flush()

if __name__ == '__main__':
    # Just some test code to check if the above functions work as expected.
    # Desired labels.
    desired_labels = [u'AUTHOR_TITLE', u'AFFILIATION', u'ABSTRACT', 
        u'AUTHOR', u'TITLE']
    map_path = '/home/msheshera/MSS/Code/Projects/meta_headers/sm-deep-metadata-extraction/feed_fwd_nw/map_dir/label_map-train-skl.pd'
    model_path = '/home/msheshera/MSS/Code/Projects/meta_headers/sm-deep-metadata-extraction/scped/mlp_untuned-fulldataset.skmodel'
    train_path = '/home/msheshera/MSS/Code/Projects/meta_headers/dataset_grotoap2/grotoap2/example_subset/npy_out/00-02_train.npy'
    
    model = load_saved_model(model_path)
    print(model)

    x, y, des_map = read_data(train_path, map_path, desired_labels)
    print(x.shape, y.shape, des_map)
