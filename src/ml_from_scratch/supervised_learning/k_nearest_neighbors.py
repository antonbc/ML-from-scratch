import numpy as np
from collections import Counter

'''
given a data point
- calculate its distance from all other data points
- get the closest K points (the value K is user defined)
- regression: get the avg of their values
- classification: get majority vote
'''


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def _euclidian_distance(self, point1, point2):
        return np.sqrt(np.sum((point1-point2) ** 2))


    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _predict_one(self, x):
        # Measure this point's distance to every saved training point.
        distances = [self._euclidian_distance(x, x_train) for x_train in self.X_train]

        # get k closest data points
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[index] for index in k_indices]

        # majority vote
        most_common_labels = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_labels
        

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x) for x in X])
