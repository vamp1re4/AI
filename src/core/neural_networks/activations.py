"""
Activation Functions Implementation and Comparison

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

Types implemented:
1. Sigmoid: σ(z) = 1/(1+e^(-z)) - Classic, but has vanishing gradients
2. Tanh: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) - Zero-centered, still vanishing gradients
3. ReLU: max(0, z) - Most popular, solves vanishing gradients for positive values
4. Leaky ReLU: max(0.01*z, z) - Fixes "dying ReLU" problem

Each function has different properties affecting gradient flow and convergence.
"""

import numpy as np

class ActivationFunctions:
    """
    Collection of activation functions and their derivatives.

    Each activation function transforms linear combinations into non-linear outputs,
    enabling neural networks to learn complex decision boundaries.
    """

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation: σ(z) = 1 / (1 + e^(-z))

        Properties:
        - Output range: (0, 1)
        - Smooth and differentiable everywhere
        - Good for binary classification probabilities
        - Suffers from vanishing gradients when |z| is large

        Args:
            z (np.ndarray): Input array

        Returns:
            np.ndarray: Sigmoid activation
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        """
        Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))

        Args:
            a (np.ndarray): Sigmoid activation values

        Returns:
            np.ndarray: Derivative values
        """
        return a * (1 - a)

    @staticmethod
    def tanh(z):
        """
        Hyperbolic tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Properties:
        - Output range: (-1, 1)
        - Zero-centered (mean ≈ 0)
        - Stronger gradients than sigmoid
        - Still suffers from vanishing gradients

        Args:
            z (np.ndarray): Input array

        Returns:
            np.ndarray: Tanh activation
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a):
        """
        Derivative of tanh: tanh'(z) = 1 - tanh(z)^2

        Args:
            a (np.ndarray): Tanh activation values

        Returns:
            np.ndarray: Derivative values
        """
        # Clip to prevent overflow
        return np.clip(1 - a**2, 0, 1)

    @staticmethod
    def relu(z):
        """
        Rectified Linear Unit: ReLU(z) = max(0, z)

        Properties:
        - Output range: [0, ∞)
        - Computationally efficient
        - No vanishing gradients for positive inputs
        - "Dying ReLU" problem: neurons can get stuck at 0

        Args:
            z (np.ndarray): Input array

        Returns:
            np.ndarray: ReLU activation
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(a):
        """
        Derivative of ReLU: ReLU'(z) = 1 if z > 0, else 0

        Args:
            a (np.ndarray): ReLU activation values (but actually needs pre-activation z)

        Returns:
            np.ndarray: Derivative values
        """
        # Note: This needs the pre-activation value z, not activation a
        # In practice, we compute this during backward pass with z
        return (a > 0).astype(float)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """
        Leaky Rectified Linear Unit: LeakyReLU(z) = max(α*z, z)

        Properties:
        - Output range: (-∞, ∞)
        - Fixes "dying ReLU" by allowing small negative gradients
        - Computationally efficient
        - No vanishing gradients

        Args:
            z (np.ndarray): Input array
            alpha (float): Slope for negative values

        Returns:
            np.ndarray: Leaky ReLU activation
        """
        return np.maximum(alpha * z, z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """
        Derivative of Leaky ReLU: LeakyReLU'(z) = α if z < 0, else 1

        Args:
            z (np.ndarray): Pre-activation values

        Returns:
            np.ndarray: Derivative values
        """
        return np.where(z < 0, alpha, 1.0)

    @staticmethod
    def get_activation(name):
        """
        Get activation function by name.

        Args:
            name (str): Activation function name

        Returns:
            callable: Activation function
        """
        activations = {
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu
        }
        return activations.get(name.lower())

    @staticmethod
    def get_activation_derivative(name):
        """
        Get activation derivative by name.

        Args:
            name (str): Activation function name

        Returns:
            callable: Derivative function
        """
        derivatives = {
            'sigmoid': ActivationFunctions.sigmoid_derivative,
            'tanh': ActivationFunctions.tanh_derivative,
            'relu': ActivationFunctions.relu_derivative,
            'leaky_relu': ActivationFunctions.leaky_relu_derivative
        }
        return derivatives.get(name.lower())