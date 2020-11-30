import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
    
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

        ### YOUR CODE HERE
        t = 0
        # np.random.seed(42)
        # self.W = np.random.rand(n_features)
        self.W = np.zeros((n_features,))
        while t < self.max_iter:
            t+=1
            g = np.mean([self._gradient(x, y) for x, y in zip(X,y)], 0) # calc gradient
            self.W += self.learning_rate*(-g) # update weights

        ### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        t = 0
        # np.random.seed(42)
        # self.W = np.random.rand(n_features)
        self.W = np.zeros((n_features,))
        while t < self.max_iter:
            t+=1
            for ix in range(0,n_samples, batch_size):
                # t+=1
                # if t>self.max_iter:
                #     break
                step = n_samples - ix if ix+batch_size > n_samples else batch_size
                g = np.mean([self._gradient(x, y) for x, y in zip(X[ix:ix+step],y[ix:ix+step])], 0)
                self.W += self.learning_rate*(-g) #update weights
        ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        t = 0
        # np.random.seed(42)
        # self.W = np.random.rand(n_features)
        self.W = np.zeros((n_features,))
        while t < self.max_iter:
            t+=1
            ix = np.random.randint(0,n_samples)
            g = self._gradient(X[ix],y[ix]) # calc gradient
            self.W += self.learning_rate*(-g) # update weights
        ### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        _g = -_y*_x/(1 + np.exp(_y*np.dot(self.W,_x)))
        return _g
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds_proba = np.array([1/(1+np.exp(np.dot(self.W,x))) for x in X])
        preds_proba = np.stack((preds_proba, 1-preds_proba),axis=-1)
        return preds_proba
        ### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = np.array([-1 if np.dot(self.W,x) < 0 else 1 for x in X])
        return preds
        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = self.predict(X)
        score = np.sum(y==preds)/n_samples*100
        return score
        ### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

