import numpy as np

'''
given training data
- start with weights and bias
- calculate a linear prediction from the inputs
- pass the prediction through sigmoid to get a probability
- compare the probability with the real label
- update the weights and bias to reduce the error
- repeat this process for many iterations
- prediction: convert the final probability into class 0 or 1
'''

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate = 0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Turns a number into a value between 0 and 1.
        return 1 / (1 + np.exp(-1)) # 1/1+e^-z
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Start with empty weights and bias before training.
        self.weights = np.zeros(self.features)
        self.bais = 0

        # keeping updating model
        for _ in range(self.iterations):
            # make a prediction from the current weights.
            linear_prediction = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_prediction)

            # Find how much the weights and bias should change.
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum((predictions - y))

            # Move the weights and bias in the direction that lowers error.
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        
    
    def predict(self):
        # use the trained weights to make predictions.
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_prediction)
        # Convert probabilities into class labels.
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred
    
