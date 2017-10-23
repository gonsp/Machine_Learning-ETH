import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

I = 176
J = 208
K = 176


class HistogramExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, offset=5):
        self.offset = offset
        self.margin = 30

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, I, J, K)

        n_elements = X.shape[0]

        X_new = None
        for i in range(0, n_elements):
            print("element", i)
            aux = []
            self.extract(X[i], aux)
            if X_new is None:
                X_new = np.zeros((n_elements, len(aux)), dtype=np.uint16)
                print(X_new.shape)
            X_new[i] = aux

        return X_new

    def extract(self, X, L, sides=3):
        I, J, K = X.shape
        i = 0
        j = 0
        for i in range(self.margin, I - self.margin, self.offset):
            for j in range(self.margin, J - self.margin, self.offset):
                L += list(X[i, j, self.margin:K - self.margin])
        if sides > 1:
            if sides == 3:
                axes = (1, 0)
            else:
                axes = (2, 0)
            X = np.rot90(X, axes=axes)
            self.extract(X, L, sides - 1)
