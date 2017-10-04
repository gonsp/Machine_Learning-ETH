from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

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
			X_new[i] = (X[i] > self.threshold).sum()/n_features

		return X_new