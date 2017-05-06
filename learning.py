"""
File contains the code which does things like training the model,
model selection and such.
"""
import os, sys, copy, errno, time
import numpy as np
from sklearn import neural_network

def train_def_model(train_x, train_y, verbose=True):
    """
    Trains an MLPClassifier on the training data passed; just the 
    default model, but with some arguments changed.
    """
    start_time = time.time()
    # Initialize classifier and train it.
    # Showing arguments which are relevant.
    mlp_clf = neural_network.MLPClassifier(
        hidden_layer_sizes=(150, ),
        activation='relu', # Default. Might tune.
        solver='sgd',
        alpha=0.0001, # Default. Might tune.
        batch_size=400,
        learning_rate='constant',
        learning_rate_init=0.001, # Default. Might tune.
        max_iter=200, # Default. Makes sense for me.
        shuffle=True, # Default. Makes sense for me.
        random_state=0,
        tol=0.0001, # Default. Might tune.
        verbose=verbose,
        momentum=0.9, # dk what this is.
        nesterovs_momentum=True) # dk what this is.
    print('Fitting model...')
    sys.stdout.flush()
    mlp_clf.fit(train_x, train_y)
    print('Fit model; loss: {:f}, solver iterations:{:d}'.format(mlp_clf.loss_, mlp_clf.n_iter_))
    fit_time = time.time()
    print('Fit model in: {:f}s'.format(fit_time-start_time))
    sys.stdout.flush()
    return mlp_clf