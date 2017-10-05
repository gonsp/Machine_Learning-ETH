from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from random import randint, seed

I = 176
J = 208
K = 176

class Sampler(BaseEstimator, TransformerMixin):

	def __init__(self, n_features=10000, sample_size=2, offset=50, random_state=None):
		self.n_features = 2*n_features
		self.sample_size = sample_size
		self.offset = offset
		seed(random_state)

	def fit(self, X, y=None):
		print("----------")
		print("Fitting")
		self.max_value = np.amax(X)
		print("Max intensity", self.max_value)
		def generate_pos():
			i = randint(self.offset, I-self.offset)
			j = randint(self.offset, J-self.offset)
			k = randint(self.offset, K-self.offset)
			return (i, j, k)
		self.positions = [generate_pos() for _ in range(self.n_features)]
		return self

	def transform(self, X, y=None):
		X = check_array(X)
		X = X.reshape(-1, I, J, K)

		n_elements = X.shape[0]

		X_new = np.zeros((X.shape[0], self.n_features))
		for i in range(0, n_elements):
			print("element", i)
			for j in range(0, self.n_features, 2):
				X_new[i][j], X_new[i][j+1] = self.sample(X[i], self.positions[j])/self.max_value

		return X_new

	def sample(self, scan, pos):
		values = []
		x, y, z = pos
		size = self.sample_size
		for i in range(max(0, x-size), min(I, x+size+1)):
			for j in range(max(0, y-size), min(J, y+size+1)):
				for k in range(max(0, z-size), min(K, z+size+1)):
					values.append(scan[i][j][k])
		values = np.asarray(values)
		mean = values.mean()
		variance = values.var()
		return (mean, variance)
