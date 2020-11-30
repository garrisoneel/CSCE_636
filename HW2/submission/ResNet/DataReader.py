import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
	"""Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""

	### YOUR CODE HERE
	fnames = os.listdir(data_dir)
	x_train = np.array([[]]).reshape(0,3072)
	y_train = np.array([])
	for fn in fnames:
		if fn.endswith("html") or fn.endswith("meta"):
			continue
		with open(os.path.join(data_dir,fn), 'rb') as fo:
			ds = pickle.load(fo, encoding='bytes')
			xtemp = np.array(ds[b'data']) #.reshape(10000,3,32,32).transpose(0,2,3,1).astype(np.uint8)
			ytemp = np.array(ds[b'labels'])

		if fn.startswith("test"):
			x_test = xtemp
			y_test = ytemp

		if fn.startswith("data"):
			x_train = np.concatenate((xtemp,x_train), axis=0)
			y_train = np.concatenate((ytemp,y_train), axis=0)
	### END CODE HERE

	return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split_index=45000):
	"""Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
	x_train_new = x_train[:split_index]
	y_train_new = y_train[:split_index]
	x_valid = x_train[split_index:]
	y_valid = y_train[split_index:]

	return x_train_new, y_train_new, x_valid, y_valid

