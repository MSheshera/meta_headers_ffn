"""
File functions which evaluate the results generated by some means.
"""
import os, sys, copy, errno, time
import pickle
import numpy as np
from sklearn import neural_network
from sklearn import metrics

def compute_f1_score(ytrue, ypred, tags):
    """
    This is the function Shankar handed. Takes arrays of labels where the 
    labels are strings.
    Input:
        ytrue: A numpy array where each element is a string label.
        ypred: A numpy array where each element is a string label.
        tags: A list of string labels.
    Returns:
        tag_level_metrics: A dictionary with precision, recall and f1 score 
        for each of the tags in 'tags'. Keyed by tag and value being a tuple (p,r,f1).
    """
    tag_level_metrics = dict()

    print('Desired tags: {}'.format(tags))
    for tag in tags:
        ids = np.where(ytrue == tag)[0]
        if len(ids) == 0: continue
        yt = np.zeros(len(ytrue))
        yp = np.zeros(len(ytrue))
        yt[ids] = 1
        yp[np.where(ypred == tag)] = 1

        tp = np.dot(yp, yt)
        fn = len(ids) - tp
        fp = sum(yp[np.setdiff1d(np.arange(len(ytrue)), ids)])

        if tp == 0:
            tag_level_metrics[tag] = (0, 0, 0)
        else:
            p = tp * 1. / (tp + fp)
            r = tp * 1. / (tp + fn)
            f1 = 2. * p * r / (p + r)
            tag_level_metrics[tag] = (p, r, f1)

    return tag_level_metrics

def evaluate_meta(ytrue, ypred, desired_map):
    """
    Find evaluation metrics for the true and predicted labels as asked by 
    Shankar (meta).
    Input:
        ytrue: A numpy array where each element is an int mapped label.
        ypred: A numpy array where each element is an int mapped label.
        desired_map: A dictionary which says what the mapping between
            a int label and a string label is.
    Returns:
        None.
    """
    # TODO: Make this return the metrics so you can use them somewhere else.
    
    # Make inverse map (make int->string)
    map_desired = dict([(v,k) for (k,v) in desired_map.items()])
    # Make an array of string labels for ytrue and ypred; compute_f1_score
    # below needs those. Not touching that function.
    ytrue_str = np.array([map_desired[y] for y in ytrue])
    ypred_str = np.array([map_desired[y] for y in ypred])
    tag_level_metrics = compute_f1_score(ytrue_str, ypred_str, desired_map.keys())
    
    accuracy = sum(ytrue == ypred) * 1. / len(ytrue)
    w_f1 = metrics.f1_score(ytrue, ypred, average='weighted')

    print('Weighted F1: {}'.format(w_f1))
    print('Accuracy: {}'.format(accuracy))
    sys.stdout.flush()

    for tag in tag_level_metrics:
        print('Precision, Recall, F1 for ' + str(tag) + ': ' + str(tag_level_metrics[tag][0]) + ', ' + str(
            tag_level_metrics[tag][1]) + ', ' + str(tag_level_metrics[tag][2]))
        sys.stdout.flush()
    # Return w_f1 so I can use it to do things with it. 
    # Mostly model selection. :-P
    return w_f1

def evaluate_skl(ytrue, ypred, desired_map):
    """
    Does the sklearn classification report evaluation.
    Input:
        ytrue: A numpy array where each element is an int mapped label.
        ypred: A numpy array where each element is an int mapped label.
        desired_map: A dictionary which says what the mapping between
            a int label and a string label is.
    Returns:
        None.
    """
    # TODO: Make this return the metrics so you can use them somewhere else.
    items = desired_map.items()
    str_labels = [tu[0]+'_'+str(tu[1]) for tu in items]
    int_labels = [tu[1] for tu in items]
    print(metrics.classification_report(ytrue, ypred, labels=int_labels, target_names=str_labels))
    sys.stdout.flush()
