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
    
def run_untuned_model(train_path, test_path, map_path, check_dir, 
    desired_labels, model_name, transform_data):
    """
    Runs a whole train and test pipeline for an untuned model.
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

def train_model(train_path, map_path, check_dir, desired_labels, model_name, 
    transform_data):
    """
    Trains a specified model and saves the model to disk.
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
    mlp_clf = learning.train_passed_model(train_x, train_y, hidden_units=210, alpha=0.1, verbose=True)

    # Save model to disk.
    model_name = os.path.join(check_dir, model_name+'.skmodel')
    gen_utils.save_model(mlp_clf, model_name)
    
    # Test on train data.
    predicted = mlp_clf.predict(train_x)
    print('\nTrain set evaluation:')
    evaluate.evaluate_meta(train_y, predicted, desired_map)

def model_selection(train_path, dev_path, map_path, check_dir, 
    desired_labels, transform_data):
    """
    Runs a model selection pipeline by going over specified hyperparameter
    values and checking performance on a dev set.
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

    # Read dev data.
    dev_x, dev_y, _ = gen_utils.read_data(dev_path, map_path, desired_labels)
    print('Dev dataset: {}'.format(dev_x.shape))
    # Map binary features to -1 or +1
    if transform_data:
        dev_x = pre_proc.transform_bin(dev_x, settings.binary_mask)
        # Normalize with learnt scaler
        dev_x = pre_proc.transform_norm(train_scaler, dev_x, 
            settings.norm_mask)
        # Map categorical features to one-hot representations.
        dev_x = pre_proc.transform_oh(train_oh_enc, dev_x, 
            settings.categorical_mask)
        print('Dev dataset transformed: {}'.format(dev_x.shape))
    sys.stdout.flush()

    # Grid search over models.
    hidden_units_range = [90, 120, 150, 180, 210]
    alpha_pow_range = range(1,7)
    model_dict = dict()
    for hidden_units in hidden_units_range:
        for alpha_pow in alpha_pow_range:
                model_name = 'mlp_hu' + str(hidden_units) + '_al' + str(alpha_pow)
                print('Model: {:s}; Hidden_units: {:d}; l2_alpha_pow: {:f}'.format(model_name, hidden_units, alpha_pow))
                l2_alpha = 10**(-alpha_pow)
                mlp_clf = learning.train_passed_model(train_x, train_y, hidden_units, l2_alpha, verbose=False)
                # Save model to disk.
                model_name = os.path.join(check_dir, model_name+'.skmodel')
                gen_utils.save_model(mlp_clf, model_name)

                # Test on dev data.
                predicted = mlp_clf.predict(dev_x)
                print('\nDev set evaluation:')
                w_f1 = evaluate.evaluate_meta(dev_y, predicted, desired_map)
                model_dict[os.path.basename(model_name)] = w_f1
                print('\n')
                sys.stdout.flush()

    # Pick best model.
    best_key = max(model_dict, key=model_dict.get)
    print('Best model: {:s}; Weighted F1: {:f}'.format(best_key, 
        model_dict[best_key]))

def run_saved_model(train_path, test_path, map_path, check_dir, 
    desired_labels, model_name, transform_data):
    """
    Run a model saved on disk.
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

    # Load model
    model_path = os.path.join(check_dir, model_name+'.skmodel')  
    mlp_clf = gen_utils.load_saved_model(model_path)
    print(mlp_clf)

    # Test on train data.
    predicted = mlp_clf.predict(train_x)
    print('\nTrain set evaluation:')
    evaluate.evaluate_meta(train_y, predicted, desired_map)
    sys.stdout.flush()

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
    parser.add_argument('-a', '--action',
            choices=['def_model', 'select_model', 'named_model','tuned_model'],
            required=True,
            help='What kind of pipeline should be run.'
            '\ndef_model: Train near default model, save model, evaluate on train on test subsets.'
            '\nselect_model: Run miltiple models and say which ones is best; saves all models to disk.'
            '\ntuned_model: Train tuned model; save model to disk; evaluate on train set.'
            '\nnamed_model: Load saved model and evaluate on train and test set.')
    cl_args = parser.parse_args()

    if cl_args.machine == 'local':
        train_path = settings.l_train_path
        dev_path = settings.l_dev_path
        test_path = settings.l_test_path
        map_path = settings.l_map_path
        check_dir = settings.l_check_dir
    elif cl_args.machine == 'S2':
        train_path = settings.s_train_path
        dev_path = settings.s_dev_path
        test_path = settings.s_test_path
        map_path = settings.s_map_path
        check_dir = settings.s_check_dir
    
    # Do whats asked for; 
    # Train many models and say which one is best.
    if cl_args.action == 'select_model':
        model_selection(train_path, dev_path, map_path, check_dir, 
            settings.desired_labels, cl_args.transform)
    # Train the almost-default sklearn model.
    elif cl_args.action == 'def_model':
        run_untuned_model(train_path, test_path, map_path, check_dir,
            settings.desired_labels, 'mlp_untuned-norm-full-p3', 
            cl_args.transform)
    # Train the tuned model.
    elif cl_args.action == 'tuned_model':
        train_model(train_path, map_path, check_dir, settings.desired_labels, 
            'mlp_hu210_al1', cl_args.transform)
    # Run a model saved on disk. Name of model needs to be specified.
    # Replace 'mlp_hu210_al1' with name of model you want to run.
    elif cl_args.action == 'named_model':
        # TODO: Allow for name of model to be passed as argument.
        # http://stackoverflow.com/a/9506443/3262406
        run_saved_model(train_path, test_path, map_path, check_dir, 
            settings.desired_labels, 'mlp_hu210_al1', cl_args.transform)

if __name__ == '__main__':
    main()