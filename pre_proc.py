"""
Code to transform the extracted features before they're learnt from.
"""
import os, sys, copy, errno, time
import numpy as np
from sklearn import preprocessing

import settings

def transform_bin(data_mat, mask):
	"""
	Given the data matrix [n_samples, n_features] and a mask [n_features]
	map the values 1,0 to 1,-1.
	This seems to be recommeded practice for neural networks.
	"""
	# Might be more inefficient than using some convoluted np code
	# but letting that pass for now.
	for idx, feat in enumerate(mask):
		if feat:
			data_mat[data_mat[:,idx]==0, idx] = -1
	return data_mat

def norm_scaler(data_mat, mask):
	"""
	Given the data matrix [n_samples, n_features] and a mask [n_features]
	learn normalization params and return a normalizer which can be used 
	on test data.
	"""
	scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
	# Learn a scaler for all values. Just throw away the values which you 
	# dont need scaling for.
	scaler.fit(data_mat)
	return scaler

def transform_norm(scaler, data_mat, mask):
	"""
	Transform data given the scaler and the mask.
	"""
	scaled = scaler.transform(data_mat)
	scaled_res = list()
	for idx, feat in enumerate(mask):
		if feat:
			scaled_res.append(scaled[:,idx].reshape(data_mat.shape[0],1))
		else:
			scaled_res.append(data_mat[:,idx].reshape(data_mat.shape[0],1))
	scaled_res = np.concatenate(scaled_res, axis=1)
	return scaled_res

def oh_encoder(data_mat, mask):
	"""
	Return a fit scaler which to use on the data.
	"""
	# Could try to specify n_values manually to be more sure but im not 
	# sure of getting that right.
	ohe = preprocessing.OneHotEncoder(n_values='auto',categorical_features=mask, sparse=False)
	ohe.fit(data_mat)
	return ohe

def transform_oh(encoder, data_mat, mask):
	"""
	Encode the categorical features specified in mask to one hot 
	representations. Make the zeros be -1s.
	"""
	# The transformer is weird because it puts all categorical
	# values in the start and the others after. This isn't clear from
	# the documentation.
	transformed = encoder.transform(data_mat)
	# Find the number columns from the start which have been set to the
	# OH encoding.
	# n_cat = np.sum(encoder.n_values_)
	# # Go over these and set the zeros to ones.
	# for idx in range(n_cat):
	# 	transformed[transformed[:,idx]==0,idx] = -1
	return transformed
	

if __name__ == '__main__':
	# Just some test code to check if the above functions work as expected.
	a = np.random.randint(1,4,(10,4))
	print(a)
	sc = norm_scaler(a,[True, False, False, True])
	print(transform_norm(sc, a, [True, False, False, True]))

	ohe = oh_encoder(a,[False, True, True, False])
	print(transform_oh(ohe, a, [False, True, True, False]))