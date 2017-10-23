import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class ModelTest(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1000):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        n_elements, n_features = X.shape

        X_new = np.zeros((n_elements, 1))
        for i in range(0, n_elements):
            X_new[i] = (X[i] > self.threshold).sum() / n_features

        return X_new
