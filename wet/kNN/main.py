
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial import distance
import numpy as np
import sys
class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)
        return self

    def predict(self, x):
        # calculate distances between all examples to x.
        distances = distance.cdist(x, self.X_train, 'euclidean')
        # sort along last axis (row) and take the K nearest.
        k_NN_indices = np.argpartition(distances, self.n_neighbors, axis = 1)[:,:self.n_neighbors]
        k_NN_labels = self.y_train[k_NN_indices]
        # most frequent label
        majority_labels = np.sign(np.sum(k_NN_labels, axis=1))
        majority_labels[majority_labels == 0] = -1
        predictions = majority_labels
        return predictions


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    k = sys.argv[1]


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
