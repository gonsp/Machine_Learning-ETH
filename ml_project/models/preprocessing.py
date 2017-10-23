from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


# Example from piazza
class Flatten(BaseEstimator, TransformerMixin):
    """Flatten"""

    def __init__(self, dim=2):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)  # Bad practice: hard-coded dimensions
        X_new = X.mean(axis=self.dim)
        X_new = X_new.reshape(X.shape[0], -1)
        return X_new
