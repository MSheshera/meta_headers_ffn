"""
File contains the code which calls stuff from elsewhere and gets
things done.
"""
import os, sys, copy, errno, time
import pickle, argparse
import numpy as np
# My modules.
from utils import gen_utils
import settings, pre_proc, learning, evaluate
    
def pipeline_ctrl(train_path, test_path, map_path, check_dir, 
    desired_labels, model_name, transform_data):
    """
    Controls the flow of the whole pipeline in general.
    """
    # Read train data.
    train_x, train_y, desired_map = gen_utils.read_data(
        train_path, map_path, desired_labels)
    print('Train dataset: {}'.format(train_x.shape))
    sys.stdout.flush()
    
    # Transform data; categorical features one-hot; rest normalized.
    # Important that the categorical features are mapped to one hot
    # vectors at the end; else masks are screwed up.
    # Map binary features to -1 or +1
    if transform_data:
        train_x = pre_proc.transform_bin(train_x, settings.binary_mask)
        # Normalize 
        train_scaler = pre_proc.norm_scaler(train_x, settings.norm_mask)
        train_x = pre_proc.transform_norm(train_scaler, train_x, 
            settings.norm_mask)
        # Map categorical features to one-hot representations.
        train_oh_enc = pre_proc.oh_encoder(train_x, 
            settings.categorical_mask)
        train_x = pre_proc.transform_oh(train_oh_enc, train_x, 
            settings.categorical_mask)
        print('Train dataset transformed: {}'.format(train_x.shape))

    # Train model
    mlp_clf = learning.train_def_model(train_x, train_y)

    # Save model to disk.
    model_name = os.path.join(check_dir, model_name+'.skmodel')
    gen_utils.save_model(mlp_clf, model_name)
    
    # Test on train data.
    predicted = mlp_clf.predict(train_x)
    print('\nTrain set evaluation:')
    evaluate.evaluate_meta(train_y, predicted, desired_map)

    # idk if using this is good but giving it a shot.
    del train_y
    del train_x

    # Read test data.
    test_x, test_y, _ = gen_utils.read_data(test_path, map_path, desired_labels)
    print('Test dataset: {}'.format(test_x.shape))
    # Map binary features to -1 or +1
    if transform_data:
        test_x = pre_proc.transform_bin(test_x, settings.binary_mask)
        # Normalize with learnt scaler
        test_x = pre_proc.transform_norm(train_scaler, test_x, 
            settings.norm_mask)
        # Map categorical features to one-hot representations.
        test_x = pre_proc.transform_oh(train_oh_enc, test_x, 
            settings.categorical_mask)
        print('Test dataset transformed: {}'.format(test_x.shape))

    # Test on test data.
    predicted = mlp_clf.predict(test_x)
    print('\nTest set evaluation:')
    evaluate.evaluate_meta(test_y, predicted, desired_map)

def main():
    """
    Parse command line arguments and call functions which do real work.
    """
    parser = argparse.ArgumentParser()
    # Where to read data from if running on my laptop or the cluster.
    parser.add_argument('-m', '--machine',
            choices=['local', 'S2'],
            required=True,
            help='Tell me if running on the local machine or on swarm2.')
    parser.add_argument('-t', '--transform',
            default=False,
            action='store_true',
            help='Should data be scaled and recoded.')
    cl_args = parser.parse_args()

    if cl_args.machine == 'local':
        train_path = settings.l_train_path
        test_path = settings.l_test_path
        map_path = settings.l_map_path
        check_dir = settings.l_check_dir
    elif cl_args.machine == 'S2':
        train_path = settings.s_train_path
        test_path = settings.s_test_path
        map_path = settings.s_map_path
        check_dir = settings.s_check_dir
    
    # Call model with appropriate data.
    pipeline_ctrl(train_path, test_path, map_path, check_dir,
        settings.desired_labels, 'mlp_untuned-norm-full-p3', cl_args.transform)

if __name__ == '__main__':
    main()