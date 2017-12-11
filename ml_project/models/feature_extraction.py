import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


n_elements = 6822
n_features = 18286


class CardiogramFeatureExtractor():
    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        X = X.reshape(n_elements, n_features)
        X_new = None

        for id in range(0, n_elements):
            features = self.extract_features(X[id])
            if X_new is None:
                X_new = np.zeros((n_elements, len(features)))

            X_new[id] = features

        return X_new

    def extract_features(self, x):
        x = np.trim_zeros(x)
        x_new = []
        X_new.append(self.extract_mean(x))
        X_new.append(self.extract_variance(x))
        X_new.append(self.extract_period(x))
        X_new.append(self.extract_max(x))
        return list(x_new)

    def extract_mean(self, x):
        return np.mean(x)

    def extract_variance(self, x):
        return np.mean(x)


I = 176
J = 208
K = 176


class HistogramGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        # Data Selection
        trainData = np.reshape(X, (-1, I, J, K))
        trainData = np.asarray(trainData)
        selectedData = trainData[:, 17:157, 20:190, 45:145]
        print("Selected Data Size: " + str(selectedData.shape))

        # Generate histograms for cubes (10*10*10 cubes) for Training
        print("Generating histograms...\n")
        histStack = []
        for i_sample in range(0, selectedData.shape[0]):
            print("Histogram of MRI:", i_sample)
            histSubStack = []
            for x in range(0, 10):
                for y in range(0, 10):
                    for z in range(0, 10):
                        cube = selectedData[i_sample,
                                            14*x:14*(x+1),
                                            17*y:17*(y+1),
                                            10*z:10*(z+1)]
                        hist, bin_edges = np.histogram(cube, bins=50)
                        hist = hist.reshape(-1, 1)
                        histSubStack = np.append(histSubStack, hist)
            histSubStack = histSubStack.reshape(-1, 1)
            if i_sample == 0:
                histStack = np.append(histStack, histSubStack).reshape(-1, 1)
            else:
                histStack = np.append(histStack, histSubStack, axis=1)
        histTrainData = histStack.T
        X_new = histTrainData

        return X_new


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

        X_new = X_new.reshape(-1, I * J * K)
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
                    length_x = \
                        min(I - 1, i + self.box_size) - \
                        max(0, i - self.box_size) + 1
                    length_y = \
                        min(J - 1, j + self.box_size) - \
                        max(0, j - self.box_size) + 1
                    length_z = \
                        min(K - 1, k + self.box_size) - \
                        max(0, k - self.box_size) + 1
                    X_new[i][j][k] /= length_x * length_y * length_z
                    # compute the mean
                    # X_new[i][j][k] /= self.max_value # normalize

        return X_new.astype(np.uint16)

    def compute_mean_matrix(self, X, M):
        for i in range(0, M.shape[0]):
            s = np.uint32(0)
            for j in range(0, self.box_size + 1):
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
