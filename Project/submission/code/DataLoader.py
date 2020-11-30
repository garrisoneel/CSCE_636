import os
import pickle
import numpy as np
import torch
from torchvision import transforms as tf

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

def load_testing_images(file):
    data = np.load(file)
    print(data.shape)
    data = torch.tensor(data.transpose(0,3,1,2)/255.0).float()
    # data = data.permute(0,3,1,2)
    print(data.shape)
    tf_norm = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    data = tf_norm(data)
    return data 

class MyDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels, training=False):
        'Initialization'
        self.labels = labels
        self.data = torch.tensor(data.reshape(data.shape[0],3,32,32)/255, dtype=torch.float32)
        # self.data = self.data.reshape(-1,3,32,32)
        # print(self.data.shape)
        # self.data = (self.data-torch.mean(self.data,dim=(-2,-1))) / (torch.std(self.data,dim=(-2,-1)))
        self.training = training
        if training:
            self.transform = tf.Compose([#tf.ToPILImage(),
                    tf.RandomHorizontalFlip(),
                    tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # internet says these are good values for cifar 10
                    tf.RandomCrop((32,32), 4, padding_mode='edge'),
                    # tf.RandomRotation(15),
                    tf.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.05, 20), value=0, inplace=False),
                    ])
        else:
            self.transform = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.data[index]
        # X = torch.reshape(self.data[index],(3,32,32))
        # X = X.permute(1,2,0)

        # X = (X-torch.mean(X, dim=(0,1)))/(torch.std(X,dim=(0,1)))
        # X = X.permute(2,0,1)
        X = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X


