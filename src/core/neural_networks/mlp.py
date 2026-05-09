"""
Multi-Layer Neural Network Implementation from Scratch

Why multi-layer networks?
- Single perceptron can only learn linearly separable functions (AND, OR, but not XOR)
- Multi-layer networks can learn complex, non-linear decision boundaries
- Hidden layers transform inputs into higher-dimensional spaces where linear separation becomes possible

XOR Gate Truth Table:
Input1 | Input2 | Output
   0   |   0    |   0
   0   |   1    |   1
   1   |   0    |   1
   1   |   1    |   0

XOR is not linearly separable - you can't draw a single straight line to separate the classes.
But a 2-layer network can learn it!
"""

import numpy as np
from .activations import ActivationFunctions
from .optimizers import get_optimizer

class MultiLayerPerceptron:
    """
    Manual implementation of a multi-layer perceptron (feedforward neural network).

    Architecture: Input -> Hidden Layer -> Output Layer
    Uses sigmoid activation for smooth gradients in backpropagation.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1,
                 hidden_activation='relu', output_activation='sigmoid', optimizer='adam'):
        """
        Initialize network with random weights.

        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons (1 for binary classification)
            learning_rate (float): Learning rate for gradient descent
            hidden_activation (str): Activation for hidden layer ('sigmoid', 'tanh', 'relu', 'leaky_relu')
            output_activation (str): Activation for output layer ('sigmoid', 'tanh', 'relu', 'leaky_relu')
            optimizer (str): Optimizer name ('sgd', 'momentum', 'rmsprop', 'adam')
        """
        # Initialize weights with appropriate scaling for activation
        if hidden_activation in ['relu', 'leaky_relu']:
            # He initialization for ReLU variants
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        else:
            # Xavier initialization for sigmoid/tanh
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)

        self.b1 = np.zeros((1, hidden_size))

        # Output layer usually uses Xavier (since we use sigmoid for binary classification)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Set activation functions
        self.hidden_activation = ActivationFunctions.get_activation(hidden_activation)
        self.hidden_activation_derivative = ActivationFunctions.get_activation_derivative(hidden_activation)
        self.output_activation = ActivationFunctions.get_activation(output_activation)
        self.output_activation_derivative = ActivationFunctions.get_activation_derivative(output_activation)

        # Store activation names for display
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation

        # Initialize optimizer
        optimizer_params = {'learning_rate': learning_rate}
        # Add any additional optimizer-specific parameters
        if optimizer == 'momentum':
            optimizer_params['momentum'] = 0.9
        elif optimizer == 'rmsprop':
            optimizer_params['beta'] = 0.9
            optimizer_params['epsilon'] = 1e-8
        elif optimizer == 'adam':
            optimizer_params['beta1'] = 0.9
            optimizer_params['beta2'] = 0.999
            optimizer_params['epsilon'] = 1e-8

        self.optimizer = get_optimizer(optimizer, **optimizer_params)

    def sigmoid(self, z):
        """
        Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))

        Properties:
        - Output range: (0, 1)
        - Smooth and differentiable
        - Good for binary classification probabilities

        Args:
            z (np.ndarray): Input array

        Returns:
            np.ndarray: Sigmoid of input
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        """
        Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))

        Args:
            a (np.ndarray): Sigmoid activation values

        Returns:
            np.ndarray: Derivative values
        """
        return a * (1 - a)

    def forward(self, X):
        """
        Forward pass through the network.

        Layer 1 (Hidden): a1 = activation(W1·X + b1)
        Layer 2 (Output): a2 = activation(W2·a1 + b2)

        Args:
            X (np.ndarray): Input matrix of shape (batch_size, input_size)

        Returns:
            tuple: (z1, a1, z2, a2) - pre-activations and activations
        """
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1  # Shape: (batch_size, hidden_size)
        a1 = self.hidden_activation(z1)    # Shape: (batch_size, hidden_size)

        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2  # Shape: (batch_size, output_size)
        a2 = self.output_activation(z2)     # Shape: (batch_size, output_size)

        return z1, a1, z2, a2

    def backward(self, X, y, a1, a2):
        """
        Backward pass - compute gradients using backpropagation.

        Uses chain rule:
        - Output error: δ2 = (a2 - y) * σ'(z2)
        - Hidden error: δ1 = δ2·W2^T * σ'(z1)

        Args:
            X (np.ndarray): Input matrix
            y (np.ndarray): True labels
            a1 (np.ndarray): Hidden layer activations
            a2 (np.ndarray): Output layer activations

        Returns:
            tuple: Gradients (dW1, db1, dW2, db2)
        """
        batch_size = X.shape[0]

        # Output layer error
        # δ2 = (a2 - y) * sigmoid_derivative(z2) = (a2 - y) * a2 * (1 - a2)
        dz2 = a2 - y  # Shape: (batch_size, output_size)
        da2_dz2 = self.sigmoid_derivative(a2)  # Shape: (batch_size, output_size)
        delta2 = dz2 * da2_dz2  # Shape: (batch_size, output_size)

        # Hidden layer error
        # δ1 = δ2·W2^T * sigmoid_derivative(z1)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(a1)  # Shape: (batch_size, hidden_size)

        # Gradients
        dW2 = np.dot(a1.T, delta2) / batch_size  # Shape: (hidden_size, output_size)
        db2 = np.sum(delta2, axis=0, keepdims=True) / batch_size  # Shape: (1, output_size)

        dW1 = np.dot(X.T, delta1) / batch_size  # Shape: (input_size, hidden_size)
        db1 = np.sum(delta1, axis=0, keepdims=True) / batch_size  # Shape: (1, hidden_size)

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        """
        Update weights using gradient descent.

        Args:
            dW1, db1, dW2, db2: Gradients from backward pass
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train_step(self, X, y):
        """
        Single training step: forward + backward + update.

        Uses binary cross-entropy loss for better gradients with sigmoid.

        Args:
            X (np.ndarray): Input batch
            y (np.ndarray): Target batch

        Returns:
            float: Binary cross-entropy loss
        """
        # Forward pass
        z1, a1, z2, a2 = self.forward(X)

        # Compute binary cross-entropy loss
        epsilon = 1e-15  # Prevent log(0)
        a2_clipped = np.clip(a2, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(a2_clipped) + (1 - y) * np.log(1 - a2_clipped))

        # Backward pass
        # For binary cross-entropy + sigmoid output, δ2 = a2 - y
        # For other output activations, we'd need different loss functions
        if self.output_activation_name == 'sigmoid':
            delta2 = a2 - y  # Shape: (batch_size, output_size)
        else:
            # For general case, multiply by activation derivative
            delta2 = (a2 - y) * self.output_activation_derivative(z2)

        # Hidden layer error: δ1 = δ2 · W2^T * activation'(z1)
        delta1 = np.dot(delta2, self.W2.T) * self.hidden_activation_derivative(z1)  # Shape: (batch_size, hidden_size)

        # Gradients
        dW2 = np.dot(a1.T, delta2) / X.shape[0]  # Shape: (hidden_size, output_size)
        db2 = np.sum(delta2, axis=0, keepdims=True) / X.shape[0]  # Shape: (1, output_size)

        dW1 = np.dot(X.T, delta1) / X.shape[0]  # Shape: (input_size, hidden_size)
        db1 = np.sum(delta1, axis=0, keepdims=True) / X.shape[0]  # Shape: (1, hidden_size)

        # Update weights using optimizer
        params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        updated_params = self.optimizer.update(params, grads)

        self.W1 = updated_params['W1']
        self.b1 = updated_params['b1']
        self.W2 = updated_params['W2']
        self.b2 = updated_params['b2']

        return loss

    def train(self, X, y, epochs=1000, batch_size=4, verbose=True):
        """
        Train the network on dataset.

        Args:
            X (np.ndarray): Training inputs
            y (np.ndarray): Training targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for mini-batch training
            verbose (bool): Print progress

        Returns:
            list: Loss history
        """
        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss

            epoch_loss /= (n_samples // batch_size)
            losses.append(epoch_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")

        return losses

    def predict(self, X):
        """
        Make predictions (threshold at 0.5 for binary classification).

        Args:
            X (np.ndarray): Input matrix

        Returns:
            np.ndarray: Binary predictions
        """
        _, _, _, a2 = self.forward(X)
        return (a2 > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X (np.ndarray): Input matrix

        Returns:
            np.ndarray: Output probabilities
        """
        _, _, _, a2 = self.forward(X)
        return a2