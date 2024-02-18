import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0")
        self.n_neighbors = n_neighbors
        self.X_memory= None
        self.y_memory = None

    def fit(self, X, y):
        # Store the training data and labels
        self.X_memory = np.copy(X)
        self.y_memory = np.copy(y)
        return self

    def predict(self, X):
        if self.X_memory is None or self.y_memory is None:
            raise Exception("The model has not been fit with training data")

        # Compute distance matrix
        distances = cdist(X, self.X_memory)

        # Adjust k if it's greater than the number of examples in memory
        k_adjusted = min(self.n_neighbors, self.X_memory.shape[0])

        # Find k nearest neighbours indices using the adjusted k value
        #np.argpartition arraging array from zero, so if for example we want 2 neighbours,
        #we must ask to partition each row such that element number 1 is in correct place
        k_nearest_indices = np.argpartition(distances, k_adjusted - 1, axis=1)[:, :k_adjusted]

        # Retrieve the labels for the k-nearest neighbors
        k_nearest_labels = self.y_memory[k_nearest_indices]

        # Sum the labels; positive sum -> majority is 1, negative sum -> majority is -1
        sum_labels = np.sum(k_nearest_labels, axis=1)
        majority_labels = np.sign(sum_labels)

        # Handling the case where sum is 0 (tie)
        majority_labels[majority_labels == 0] = -1

        # The majority_labels now contains the predicted label for each point in X
        predictions = majority_labels

        return predictions