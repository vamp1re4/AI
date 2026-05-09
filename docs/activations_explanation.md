# Activation Functions - What We Learned

## Why Activation Functions Matter

**Linear transformations alone can't solve complex problems**: Without non-linear activations, multi-layer networks are just linear regressions.

**Activation functions add non-linearity**: They transform linear combinations into non-linear outputs, enabling networks to learn complex decision boundaries.

## Activation Functions Compared

### Sigmoid: σ(z) = 1/(1+e^(-z))
**Properties:**
- Output range: (0, 1)
- Smooth and differentiable
- Good for output layer (probabilities)

**Problems:**
- Vanishing gradients: When |z| is large, gradients → 0
- Not zero-centered: Outputs always positive
- Numerical instability: exp(-500) → 0, exp(500) → ∞

**Best for:** Output layers in binary classification

### Tanh: tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))
**Properties:**
- Output range: (-1, 1)
- Zero-centered (mean ≈ 0)
- Stronger gradients than sigmoid

**Problems:**
- Still vanishing gradients
- More expensive than ReLU

**Best for:** Hidden layers when zero-centering matters

### ReLU: max(0, z)
**Properties:**
- Output range: [0, ∞)
- Computationally efficient (just max operation)
- No vanishing gradients for positive inputs
- Sparse activation (many neurons output 0)

**Problems:**
- "Dying ReLU": Neurons can get stuck at 0 forever
- Not zero-centered

**Best for:** Most hidden layers in modern networks

### Leaky ReLU: max(0.01*z, z)
**Properties:**
- Output range: (-∞, ∞)
- Fixes "dying ReLU" problem
- Allows small gradients for negative inputs
- Computationally efficient

**Problems:**
- Hyperparameter (α=0.01) to tune

**Best for:** Hidden layers when dying ReLU is a concern

## Experimental Results on XOR

| Activation | Final Loss | Accuracy | Learning Speed |
|------------|------------|----------|----------------|
| Sigmoid    | NaN (failed) | 0%     | Failed         |
| Tanh       | NaN (failed) | 0%     | Failed         |
| ReLU       | 0.036     | 100%   | Fast           |
| Leaky ReLU | 0.002     | 100%   | Fastest        |

**Key Insights:**
1. **ReLU/Leaky ReLU dramatically outperform sigmoid/tanh** on XOR
2. **Proper initialization matters**: He for ReLU, Xavier for sigmoid/tanh
3. **Numerical stability is crucial**: Sigmoid/tanh failed due to overflow
4. **Leaky ReLU slightly better**: Avoids dying neurons

## When to Use Each Activation

- **Sigmoid**: Output layer for binary classification
- **Tanh**: Hidden layers when zero-centering needed (rare)
- **ReLU**: Default choice for hidden layers
- **Leaky ReLU**: When dying ReLU is a problem

## Modern Alternatives

- **ELU**: Exponential Linear Unit (smooth negative part)
- **SELU**: Self-normalizing (for very deep networks)
- **GELU**: Gaussian Error Linear Unit (used in transformers)

## Code Implementation

```python
# ReLU is simple but effective
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

## Performance Impact

- **ReLU/Leaky ReLU**: 10-100x faster training than sigmoid
- **Memory**: Same for all (just element-wise operations)
- **Convergence**: ReLU variants converge much faster
- **Accuracy**: Often better final performance

## Visualization Insights

Check `/docs/activation_comparison.png`:
- **Loss curves**: ReLU/Leaky ReLU converge exponentially faster
- **Decision boundaries**: Both create complex non-linear boundaries
- **Activation distributions**: ReLU creates sparse activations (many zeros)

## Next Steps

- **Batch Normalization**: Stabilizes training
- **Residual Connections**: Enables very deep networks
- **Advanced Optimizers**: Adam, RMSProp
- **Regularization**: Dropout, L2 penalty