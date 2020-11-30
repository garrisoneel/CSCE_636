#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author:
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
    
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        y = np.zeros((n_samples,self.k,))
        for ix,label in enumerate(labels):
            y[ix,int(label)] = 1
        # np.random.seed(42)
        # self.W = np.random.rand(self.k,n_features)
        self.W = np.zeros((self.k,n_features))
        t = 0

        while t < self.max_iter:
            t+=1
            for ix in range(0,n_samples, batch_size):
                # t+=1
                # if t>self.max_iter:
                #     break
                step = n_samples - ix if ix+batch_size > n_samples else batch_size
                grads = np.array([self._gradient(X[k], y[k]) for k in range(ix,ix+step)])
                avg_grad = np.mean(grads, axis=0)
                self.W -= self.learning_rate*avg_grad
                # if np.linalg.norm(avg_grad*1./batch_size) < 0.0005:
                #     break

        ### END YOUR CODE

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        n_features,*other = _x.shape
        v = np.matmul(self.W, _x)
        q = self.softmax(v)
        sfmx = np.reshape(q-_y,(self.k,1))
        _g = np.matmul(sfmx, np.reshape(_x,(n_features,1)).T)
        return _g
        ### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

        ### YOUR CODE HERE
        _softmax = np.exp(x)/np.sum(np.exp(x))
        return _softmax
        ### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        proba = np.array([self.softmax(np.matmul(self.W,X[k])) for k in range(n_samples)])
        preds = np.argmax(proba, axis=1)
        return preds
        ### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = self.predict(X)
        corr = (labels == preds)
        acc = np.sum(corr)/n_samples*100
        return acc
        ### END YOUR CODE

