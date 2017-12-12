import sys

from ml_project.models.feature_extraction import CardiogramFeatureExtractor, n_features

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


class MRIVisualizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        I = 176
        J = 208
        K = 176
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
                image = np.rot90(X[id], axes=(0, 2))[height]
                # image = np.zeros((I, K))
                # for i in range(0, I):
                # 	for k in range(0, K):
                # 		image[I-1-i][k] = X[id][i][height][k]
                # image = np.rot90(image, axes=(0,1))
                plt.imshow(image, cmap='gray')
                plt.draw()
                plt.pause(0.001)

    def transform(self, X, y):
        pass

class CardiogramVisualizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X = X.reshape(-1, n_features)

        feature_extractor = CardiogramFeatureExtractor()
        feature_extractor.fit(X)
        extracted_features = feature_extractor.transform(X)

        plt.ion()
        plt.show()

        fig_size = plt.rcParams["figure.figsize"]
         
        print("Current size:", fig_size)
         
        # Set figure width to 12 and height to 9
        fig_size[0] = 14
        fig_size[1] = 5
        plt.rcParams["figure.figsize"] = fig_size

        first = True
        id = 0
        while True:
            action = True

            if not first:
                c = sys.stdin.read(1)
                if c == 'd':
                    id += 1
                elif c == 'a':
                    id -= 1
                else:
                    action = False

            first = False

            if action:
                plt.clf()
                print("id:", id)
                print("class: ", y[id])
                print("features: ", extracted_features[id])
                plt.plot(np.trim_zeros(X[id]))
                plt.show()
                plt.pause(0.001)



    def transform(self, X, y):
        pass
