"""
Optimization Algorithms Implementation

Gradient descent variants dramatically improve training:
1. SGD: Basic gradient descent (what we've used)
2. Momentum: Accelerates convergence, reduces oscillations
3. RMSProp: Adaptive learning rates per parameter
4. Adam: Combines momentum + RMSProp (most popular)

Each optimizer has different convergence properties and hyperparameter requirements.
"""

import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms.

    All optimizers implement the same interface:
    - update(params, grads)
    - Maintain internal state (momentum, adaptive rates, etc.)
    """

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, params, grads):
        """
        Update parameters using gradients.

        Args:
            params (dict): Parameter dictionary {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            grads (dict): Gradient dictionary {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        Returns:
            dict: Updated parameters
        """
        raise NotImplementedError

    def set_learning_rate(self, learning_rate: float):
        """Update the current learning rate."""
        self.learning_rate = learning_rate

    def _apply_weight_decay(self, param, grad):
        if self.weight_decay and self.weight_decay > 0.0:
            return grad + self.weight_decay * param
        return grad

class SGD(Optimizer):
    """
    Stochastic Gradient Descent - Basic optimizer.

    Simply: params = params - learning_rate * grads

    Properties:
    - Simple and reliable
    - Can get stuck in local minima
    - Oscillates in ravines (valleys with steep sides)
    - Learning rate critical (too high = diverge, too low = slow)
    """

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)

    def update(self, params, grads):
        updated_params = {}
        for key in params:
            grad = self._apply_weight_decay(params[key], grads[key])
            updated_params[key] = params[key] - self.learning_rate * grad
        return updated_params

class SGDMomentum(Optimizer):
    """
    SGD with Momentum - Accelerates convergence.

    Maintains velocity vector: v = β*v + (1-β)*grad
    Then: params = params - learning_rate * v

    Properties:
    - Accelerates in consistent gradient directions
    - Reduces oscillations in ravines
    - β typically 0.9 (momentum coefficient)
    - Helps escape local minima
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        updated_params = {}

        for key in params:
            grad = self._apply_weight_decay(params[key], grads[key])
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])

            self.velocity[key] = self.momentum * self.velocity[key] + (1 - self.momentum) * grad
            updated_params[key] = params[key] - self.learning_rate * self.velocity[key]

        return updated_params

class RMSProp(Optimizer):
    """
    RMSProp - Adaptive learning rates per parameter.

    Maintains moving average of squared gradients: s = β*s + (1-β)*grad²
    Then: params = params - learning_rate * grad / sqrt(s + ε)

    Properties:
    - Adapts learning rate per parameter
    - Good for sparse gradients
    - Helps with vanishing/exploding gradients
    - β typically 0.9, ε=1e-8
    """

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.squared_grad = {}

    def update(self, params, grads):
        updated_params = {}

        for key in params:
            grad = self._apply_weight_decay(params[key], grads[key])
            if key not in self.squared_grad:
                self.squared_grad[key] = np.zeros_like(params[key])

            self.squared_grad[key] = self.beta * self.squared_grad[key] + (1 - self.beta) * (grad ** 2)
            adaptive_lr = self.learning_rate / (np.sqrt(self.squared_grad[key]) + self.epsilon)
            updated_params[key] = params[key] - adaptive_lr * grad

        return updated_params

class Adam(Optimizer):
    """
    Adam - Adaptive Moment Estimation (most popular optimizer).

    Combines momentum + RMSProp:
    - First moment (momentum): m = β1*m + (1-β1)*grad
    - Second moment (RMSProp): v = β2*v + (1-β2)*grad²
    - Bias correction: m̂ = m/(1-β1^t), v̂ = v/(1-β2^t)
    - params = params - learning_rate * m̂ / sqrt(v̂ + ε)

    Properties:
    - Best of both worlds (momentum + adaptive rates)
    - Default choice for most deep learning
    - β1=0.9, β2=0.999, ε=1e-8
    - Little hyperparameter tuning needed
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMSProp)
        self.t = 0   # Time step

    def update(self, params, grads):
        self.t += 1
        updated_params = {}

        for key in params:
            grad = self._apply_weight_decay(params[key], grads[key])
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_params

def get_optimizer(name, **kwargs):
    """
    Get optimizer by name.

    Args:
        name (str): Optimizer name ('sgd', 'momentum', 'rmsprop', 'adam')
        **kwargs: Optimizer-specific parameters

    Returns:
        Optimizer: Optimizer instance
    """
    optimizers = {
        'sgd': SGD,
        'momentum': SGDMomentum,
        'rmsprop': RMSProp,
        'adam': Adam
    }

    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    return optimizers[name.lower()](**kwargs)