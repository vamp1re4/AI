# Multi-Layer Neural Networks - What We Learned

## Why Multi-Layer Networks Matter

**Single perceptrons are limited**: They can only learn linearly separable functions like AND, OR, but not XOR.

**XOR Gate Problem**:
- Input [0,0] → Output 0
- Input [0,1] → Output 1
- Input [1,0] → Output 1
- Input [1,1] → Output 0

XOR cannot be separated by a single straight line in 2D space. You need a "curved" decision boundary.

**Solution**: Hidden layers transform inputs into higher-dimensional spaces where linear separation becomes possible.

## Network Architecture

```
Input Layer (2 neurons) → Hidden Layer (8 neurons) → Output Layer (1 neuron)
```

- **Input layer**: Takes the 2D input [x1, x2]
- **Hidden layer**: Learns intermediate representations using sigmoid activation
- **Output layer**: Produces final prediction using sigmoid for probabilities

## Forward Pass Mathematics

1. **Hidden layer computation**:
   ```
   z1 = X · W1 + b1
   a1 = σ(z1)  # Sigmoid activation
   ```

2. **Output layer computation**:
   ```
   z2 = a1 · W2 + b2
   a2 = σ(z2)  # Sigmoid activation
   ```

Where σ(z) = 1 / (1 + e^(-z)) is the sigmoid function.

## Learning: Backpropagation

**Key Insight**: We use binary cross-entropy loss instead of MSE for better gradient flow.

**Loss Function**:
```
L = -[y · log(a2) + (1-y) · log(1-a2)]
```

**Gradient Computation** (simplified for sigmoid + cross-entropy):
- Output error: δ2 = a2 - y
- Hidden error: δ1 = δ2 · W2^T · σ'(z1)

**Weight Updates**:
```
W2 = W2 - learning_rate · (a1^T · δ2) / batch_size
W1 = W1 - learning_rate · (X^T · δ1) / batch_size
```

## Why Cross-Entropy Beats MSE

**MSE with Sigmoid**: Gradients vanish when predictions are confident (close to 0 or 1).

**Cross-Entropy**: Provides strong gradients regardless of prediction confidence.

Our loss dropped from 0.738 to 0.036 - a 95% reduction!

## Decision Boundary Visualization

Check `/docs/mlp_*.png`:
- **Untrained**: Random predictions around 0.5
- **Trained**: Complex curved boundary that perfectly separates XOR classes
- **Loss curve**: Exponential decay showing effective learning

## Key Takeaways

1. **Hidden layers enable non-linear learning**
2. **Activation functions add non-linearity**
3. **Loss function choice dramatically affects convergence**
4. **Backpropagation chains gradients through layers**
5. **Multi-layer networks can learn arbitrary complex functions**

## Performance Results

- **Training time**: 5000 epochs
- **Final loss**: 0.036 (binary cross-entropy)
- **Accuracy**: 100% on XOR gate
- **Parameters**: 33 total (17 for W1/b1, 16 for W2/b2)

## Next Steps

- **Activation functions**: ReLU for faster training
- **Deep networks**: Multiple hidden layers
- **Optimization**: Adam, momentum
- **Regularization**: Dropout, L2 penalty
- **Convolutional layers**: For images
- **Recurrent layers**: For sequences

## Code Structure

- `src/core/neural_networks/mlp.py` - Complete MLP implementation
- `tests/test_mlp.py` - XOR gate test
- `src/utils/visualize_mlp.py` - Decision boundary visualization
- `docs/` - Generated plots and explanations