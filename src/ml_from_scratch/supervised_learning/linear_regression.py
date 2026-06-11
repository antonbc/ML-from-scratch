import numpy as np

class LinearRegressionFromScratchClosed:

    def __init__(self):
        # will hold the slope coefficients (beta_1 ... beta_p) after fit() is called
        self.slopes = None
        # will hold the intercept (beta_0) after fit() is called
        self.intercept = 0.0

    def fit(self, features, targets):
        # features : 2D array of shape (n_samples, n_predictors)
        #            each row is one observation
        #            each column is one predictor
        # targets  : 1D array of shape (n_samples,)
        #            the response values we are trying to predict
        features = np.array(features)
        targets = np.array(targets)

        # prepend a column of ones so the intercept is solved automatically
        # as the first element of the solution vector
        # shape goes from (n_samples, n_predictors)
        #               to (n_samples, n_predictors + 1)
        design_matrix = np.c_[np.ones((features.shape[0], 1)), features]

        # normal equation: beta = (X^T X)^-1 X^T y
        # @ is the matrix multiplication operator (same as np.matmul)
        # solution shape: (n_predictors + 1,) -> [beta_0, beta_1, ..., beta_p]
        solution = (
            np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ targets
        )

        # first element is the intercept beta_0
        self.intercept = solution[0]
        # remaining elements are the slope coefficients beta_1 ... beta_p
        self.slopes = solution[1:]

    def predict(self, features):
        # features : 2D array of shape (n_samples, n_predictors)
        features = np.array(features)

        # y_hat = X * slopes + intercept
        # features @ self.slopes -> dot product of features with slope coefficients
        # + self.intercept       -> adds beta_0 to every prediction
        return features @ self.slopes + self.intercept


# y = mx + b or B_0 + B_1x_1 + B_2x_2 + ....
class SimpleLinearRegressionFromScratchGradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.b = 0

    # X = points on the graph
    def gradient_descent(self, X, y):
        partial_derivative_m = 0
        partial_derivative_b = 0

        n = len(X)

        for i in range(n):
            x_i = X[i]
            y_i = y[i]

            partial_derivative_m += -(2 / n) * x_i * (y_i - ((self.m * x_i) + self.b))
            partial_derivative_b += -(2 / n) * (y - ((self.m * x_i) + self.b))

        self.m = self.m - self.learning_rate * partial_derivative_m
        self.b = self.b - self.learning_rate * partial_derivative_b

    def fit(self, X, y):
        for _ in range(self.iterations):
            self.gradient_descent(X, y)

    def predict(self, X):
        predictions = []

        for x_i in X:
            # y = mx + b
            y_pred = (self.m * x_i) + self.b
            predictions.append(y_pred)
        return predictions


# multi input
class LinearRegressionFromScratchGradientDescent:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.b = 0

    def gradient_descent(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])

        partial_derivative_weights = np.zeros(n_features)
        partial_derivative_b = 0

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            y_pred = np.dot(self.weights, x_i) + self.b
            error = y_i - y_pred

            for j in range(n_features):
                partial_derivative_weights[j] += -(2 / n_samples) * x_i[j] * error

            partial_derivative_b += -(2 / n_samples) * error

        self.weights -= self.learning_rate * partial_derivative_weights
        self.b -= self.learning_rate * partial_derivative_b

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_features = len(X[0])

        self.weights = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iterations):
            self.gradient_descent(X, y)

        return self

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x_i in X:
            y_pred = np.dot(self.weights, x_i) + self.b
            predictions.append(y_pred)

        return np.array(predictions)