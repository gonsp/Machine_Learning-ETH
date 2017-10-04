import sys
from sklearn.utils.validation import check_array
from matplotlib import pyplot as plt
import numpy as np

class MRIVisualizer:

	def __init__(self):
		pass

	def fit(self, X, y):
		I = 176
		J = 208
		K = 176
		X = check_array(X)
		X = X.reshape(-1, I, J, K)

		id = 0
		height = 100

		plt.ion()
		plt.show()

		while True:
			c = sys.stdin.read(1)
			action = True
			if c == 'w':
				height += 5
				print("height: ", height)
			elif c == 's':
				height -= 5
				print("height: ", height)
			elif c == 'd':
				id += 1
				print("id: ", id)
			elif c == 'a':
				id -= 1
				print("id: ", id)
			else:
				action = False

			if action:
				print("age: ", y[id])
				image = np.rot90(X[id], axes=(0,2))[height]
				# image = np.zeros((I, K))
				# for i in range(0, I):
				# 	for k in range(0, K):
				# 		image[I-1-i][k] = X[id][i][height][k]
				# image = np.rot90(image, axes=(0,1))
				plt.imshow(image, cmap='gray')
				plt.draw()
				plt.pause(0.001)
