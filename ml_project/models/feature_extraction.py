from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from random import randint, seed

I = 176
J = 208
K = 176

class MeanTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, box_size=5):
		self.box_size = box_size

	def fit(self, X, y=None):
		print("----------")
		print("Fitting")
		self.max_value = np.amax(X)
		print("Max intensity", self.max_value)
		return self

	def transform(self, X, y=None):
		X = check_array(X)
		X = X.reshape(-1, I, J, K)

		X_new = np.zeros(X.shape, dtype=np.uint16) 

		for i in range(0, X.shape[0]):
			print("Computing mean matrix for element", i)
			X_new[i] = self.compute_mean_matrix_3D(X[i])
			print("Mean matrix computed")

		X_new = X_new.reshape(-1, I*J*K)
		return X_new


	def compute_mean_matrix_3D(self, X):
		X_new = np.zeros(X.shape, dtype=np.uint32)

		for i in range(0, X.shape[0]):
			self.compute_mean_matrix(X[i], X_new[i])

			aux1 = np.rot90(X_new[i], axes=(0, 1))
			aux2 = np.zeros(aux1.shape)

			self.compute_mean_matrix(aux1, aux2)

			X_new[i] = np.rot90(aux2, axes=(1, 0))

		aux1 = np.rot90(X_new, axes=(0, 2))
		aux2 = np.zeros(aux1.shape)

		for i in range(0, aux1.shape[0]):
			self.compute_mean_matrix(aux1[i], aux2[i])

		X_new[:] = np.rot90(aux2, axes=(2, 0))

		for i in range(0, I):
			for j in range(0, J):
				for k in range(0, K):
					length_x = min(I-1, i+self.box_size) - max(0, i-self.box_size) + 1
					length_y = min(J-1, j+self.box_size) - max(0, j-self.box_size) + 1
					length_z = min(K-1, k+self.box_size) - max(0, k-self.box_size) + 1
					X_new[i][j][k] /= length_x * length_y * length_z # compute the mean
					# X_new[i][j][k] /= self.max_value # normalize
		
		return X_new.astype(np.uint16)

	def compute_mean_matrix(self, X, M):
		for i in range(0, M.shape[0]):
			s = np.uint32(0)
			for j in range(0, self.box_size+1):
				s += X[i][j]
			M[i][0] = s
			
			for j in range(1, M.shape[1]):
				old_pos = j - self.box_size - 1
				if old_pos >= 0:
					s -= X[i][old_pos]
				new_pos = j + self.box_size
				if new_pos < M.shape[1]:
					s += X[i][new_pos]
				M[i][j] = s
