import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-1)) # 1/1+e^-z

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate = 0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(self.features)
        self.bais = 0

        for _ in range(self.iterations):
            linear_prediction = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_prediction)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum((predictions - y))

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        
    
    def predict(self):
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_prediction)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred
    