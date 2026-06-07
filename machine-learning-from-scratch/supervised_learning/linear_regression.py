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
        targets  = np.array(targets)

        # prepend a column of ones so the intercept is solved automatically
        # as the first element of the solution vector
        # shape goes from (n_samples, n_predictors)
        #               to (n_samples, n_predictors + 1)
        design_matrix = np.c_[np.ones((features.shape[0], 1)), features]

        # normal equation: beta = (X^T X)^-1 X^T y
        # @ is the matrix multiplication operator (same as np.matmul)
        # solution shape: (n_predictors + 1,) -> [beta_0, beta_1, ..., beta_p]
        solution = (
            np.linalg.inv(design_matrix.T @ design_matrix)
            @ design_matrix.T
            @ targets
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


class LinearRegressionFromScratchGradient:

    def __init__(self, learning_rate, n_iterations):
        # alpha - step size for each gradient update
        # too large  -> overshoots the minimum and diverges
        # too small  -> converges correctly but very slowly
        self.learning_rate = learning_rate

        # number of times we loop over the data updating the parameters
        self.n_iterations = n_iterations

        # slope coefficients beta_1 ... beta_p, initialised to zeros in fit()
        self.slopes = None

        # intercept beta_0, initialised to zero in fit()
        self.intercept = None

    def fit(self, features, targets):
        # features : 2D array of shape (n_samples, n_predictors)
        # targets  : 1D array of shape (n_samples,)
        n_samples, n_predictors = features.shape

        # initialise all slope coefficients to zero before training begins
        self.slopes    = np.zeros(n_predictors)
        # initialise intercept to zero
        self.intercept = 0.0

        for _ in range(self.n_iterations):

            # forward pass: compute current predictions using present parameters
            # predictions = X * slopes + intercept,  shape: (n_samples,)
            predictions = np.dot(features, self.slopes) + self.intercept

            # residuals = y_hat - y (how wrong each prediction is)
            residuals = predictions - targets

            # gradient of RSS with respect to slopes
            # dL/d(slopes) = (1/n) * X^T * residuals
            # features.T is (n_predictors, n_samples)
            # residuals  is (n_samples,)
            # result     is (n_predictors,) - one gradient value per slope
            slopes_gradient = (1 / n_samples) * np.dot(features.T, residuals)

            # gradient of RSS with respect to intercept
            # dL/d(intercept) = (1/n) * sum(residuals)
            intercept_gradient = (1 / n_samples) * np.sum(residuals)

            # gradient descent update rule: parameter = parameter - alpha * gradient
            # move each parameter a small step opposite its gradient direction
            self.slopes    = self.slopes    - self.learning_rate * slopes_gradient
            self.intercept = self.intercept - self.learning_rate * intercept_gradient

    def predict(self, features):
        # features : 2D array of shape (n_samples, n_predictors)
        # y_hat = X * slopes + intercept
        return np.dot(features, self.slopes) + self.intercept