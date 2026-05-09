"""
Perceptron Implementation from Scratch

A perceptron is the simplest form of a neural network - a single neuron that can learn to classify inputs.
It takes multiple inputs, multiplies them by weights, adds a bias, and applies an activation function.

Mathematical foundation:
- Input: x = [x1, x2, ..., xn] (vector of n features)
- Weights: w = [w1, w2, ..., wn] (vector of n weights)
- Bias: b (scalar offset)
- Output: y = activation(sum(x_i * w_i) + b)

For binary classification, we use step function: 1 if sum >= 0, else 0
"""

import numpy as np

class Perceptron:
    """
    Manual implementation of a single perceptron (neuron).

    Attributes:
        weights (np.ndarray): Weight vector of shape (n_features,)
        bias (float): Bias term
        learning_rate (float): Learning rate for gradient descent
    """

    def __init__(self, n_features, learning_rate=0.01):
        """
        Initialize perceptron with random weights and zero bias.

        Args:
            n_features (int): Number of input features
            learning_rate (float): Step size for weight updates
        """
        # Initialize weights randomly between -1 and 1
        self.weights = np.random.uniform(-1, 1, n_features)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activation(self, z):
        """
        Step activation function.
        Returns 1 if z >= 0, else 0.

        Args:
            z (float): Weighted sum + bias

        Returns:
            int: Binary output (0 or 1)
        """
        return 1 if z >= 0 else 0

    def forward(self, x):
        """
        Forward pass: compute prediction for input x.

        Args:
            x (np.ndarray): Input vector of shape (n_features,)

        Returns:
            int: Predicted class (0 or 1)
        """
        # Compute weighted sum: dot product of weights and inputs + bias
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)

    def train_step(self, x, y_true):
        """
        Single training step using perceptron learning rule.

        Args:
            x (np.ndarray): Input vector of shape (n_features,)
            y_true (int): True label (0 or 1)

        Returns:
            int: Prediction error (y_pred - y_true)
        """
        # Get prediction
        y_pred = self.forward(x)

        # Compute error
        error = y_pred - y_true

        # Update weights and bias using gradient descent
        # Weight update: w_new = w_old - learning_rate * error * x
        # Bias update: b_new = b_old - learning_rate * error
        self.weights -= self.learning_rate * error * x
        self.bias -= self.learning_rate * error

        return error

    def train(self, X, y, epochs=100, verbose=True):
        """
        Train the perceptron on a dataset.

        Args:
            X (np.ndarray): Training inputs of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
            epochs (int): Number of training epochs
            verbose (bool): Whether to print progress

        Returns:
            list: Training errors per epoch
        """
        errors_per_epoch = []

        for epoch in range(epochs):
            total_error = 0

            # Train on each sample
            for x_i, y_i in zip(X, y):
                error = self.train_step(x_i, y_i)
                total_error += abs(error)

            errors_per_epoch.append(total_error)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Total error = {total_error}")

        return errors_per_epoch

    def predict(self, X):
        """
        Predict labels for multiple inputs.

        Args:
            X (np.ndarray): Input matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions of shape (n_samples,)
        """
        predictions = []
        for x in X:
            predictions.append(self.forward(x))
        return np.array(predictions)