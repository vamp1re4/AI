# Perceptron Implementation - What We Learned

## What is a Perceptron?
A perceptron is the building block of neural networks - it's like a single artificial neuron that can learn to classify inputs. Think of it as a decision-making unit that combines multiple inputs with learned weights to make a binary choice (yes/no, true/false).

## How It Works Mathematically

### Forward Pass
1. **Inputs**: Vector `x = [x1, x2, ..., xn]` (n features)
2. **Weights**: Vector `w = [w1, w2, ..., wn]` (learned parameters)
3. **Bias**: Scalar `b` (learned offset)
4. **Weighted Sum**: `z = w·x + b = Σ(w_i * x_i) + b`
5. **Activation**: `y = step(z)` where step(z) = 1 if z ≥ 0, else 0

### Learning (Gradient Descent)
For each training example `(x, y_true)`:
1. Predict: `y_pred = perceptron.forward(x)`
2. Compute error: `error = y_pred - y_true`
3. Update weights: `w = w - learning_rate * error * x`
4. Update bias: `b = b - learning_rate * error`

## Why This Works
- **Linearity**: The perceptron finds a linear decision boundary that separates classes
- **Gradient Descent**: Moves weights in the direction that reduces error
- **Convergence**: For linearly separable data (like AND gate), it will eventually find the correct boundary

## Key Insights
1. **Weights represent importance**: Higher weight means that input feature matters more
2. **Bias shifts the boundary**: Allows the decision line to not pass through origin
3. **Learning rate controls speed**: Too high = unstable, too low = slow convergence
4. **Perceptrons are limited**: Can only learn linearly separable functions

## AND Gate Example
The perceptron learned: `output = step(0.12*x1 + 0.30*x2 - 0.30)`

This creates a decision boundary where:
- Points above the line: Class 1 (True)
- Points below the line: Class 0 (False)

## Visual Learning
Check the plots in `/docs/` to see how the decision boundary evolved from random (initial) to correct (final).

## Next Steps
- Multi-layer networks (can learn non-linear functions)
- Different activation functions (sigmoid, ReLU)
- Backpropagation for deep networks
- Loss functions beyond simple error

## Code Structure
- `Perceptron` class in `src/core/neural_networks/perceptron.py`
- Test script in `tests/test_perceptron.py`
- Visualization in `src/utils/visualize_perceptron.py`