# Optimization Algorithms - What We Learned

## Why Optimization Matters

**Basic gradient descent is slow and unreliable**: Simple parameter updates often get stuck or oscillate wildly.

**Advanced optimizers make training 10-100x faster**: Modern optimization algorithms adapt learning rates, add momentum, and stabilize convergence.

## Optimizer Algorithms Compared

### SGD (Stochastic Gradient Descent)
**Update rule**: `θ = θ - η · ∇L(θ)`

**Properties:**
- Simple and reliable baseline
- Learning rate critical (too high = diverge, too low = slow)
- Can oscillate in ravines (narrow valleys)
- Gets stuck in local minima

**Best for:** Simple problems, when you want full control

### SGD with Momentum
**Update rules**:
```
v = β·v + (1-β)·∇L(θ)
θ = θ - η·v
```

**Properties:**
- Accelerates in consistent gradient directions
- Reduces oscillations in ravines
- β = 0.9 (momentum coefficient) works well
- Helps escape shallow local minima

**Best for:** Problems with ravines or when SGD oscillates

### RMSProp (Adaptive Learning Rates)
**Update rules**:
```
s = β·s + (1-β)·∇L(θ)²
θ = θ - η·∇L(θ) / sqrt(s + ε)
```

**Properties:**
- Adapts learning rate per parameter
- Excellent for sparse gradients
- Prevents exploding gradients
- β = 0.9, ε = 1e-8 standard values

**Best for:** Recurrent networks, sparse data

### Adam (Most Popular)
**Update rules**:
```
m = β1·m + (1-β1)·∇L(θ)          # First moment (momentum)
v = β2·v + (1-β2)·∇L(θ)²         # Second moment (RMSProp)
m̂ = m / (1-β1^t)                  # Bias correction
v̂ = v / (1-β2^t)                  # Bias correction
θ = θ - η·m̂ / sqrt(v̂ + ε)
```

**Properties:**
- Combines momentum + adaptive rates
- Default choice for most deep learning
- Little hyperparameter tuning needed
- β1 = 0.9, β2 = 0.999, ε = 1e-8

**Best for:** Most modern neural networks

## Experimental Results on XOR

| Optimizer | Final Loss | Convergence Speed | Stability |
|-----------|------------|-------------------|-----------|
| SGD       | 0.005     | Medium           | Good      |
| Momentum  | 0.005     | Medium           | Better    |
| RMSProp   | 0.000     | Fastest          | Excellent |
| Adam      | 0.001     | Fast             | Very Good |

**Key Insights:**
1. **RMSProp achieved best performance**: Lowest final loss, fastest convergence
2. **All modern optimizers beat basic SGD**: Momentum/Adam/RMSProp all better
3. **Adaptive methods excel**: RMSProp/Adam adapt learning rates automatically
4. **Momentum reduces oscillations**: Helps with ravine landscapes

## When to Use Each Optimizer

- **SGD**: Educational purposes, simple problems, full control needed
- **Momentum**: When SGD oscillates, ravine-like loss landscapes
- **RMSProp**: Recurrent networks, sparse gradients, stable training needed
- **Adam**: Default choice, most problems, minimal tuning

## Implementation Details

**Parameter initialization matters**:
```python
# Different optimizers need different learning rates
learning_rates = {
    'sgd': 0.1,
    'momentum': 0.1,
    'rmsprop': 0.01,  # Much smaller for adaptive methods
    'adam': 0.01
}
```

**Bias correction in Adam**:
- Prevents initial bias toward zero
- m̂ = m/(1-β1^t) corrects momentum bias
- v̂ = v/(1-β2^t) corrects RMS bias

## Performance Impact

- **Convergence speed**: Adam/RMSProp 5-10x faster than SGD
- **Memory usage**: All similar (just storing gradients/moments)
- **Hyperparameter sensitivity**: Adam least sensitive
- **Final accuracy**: Often slightly better with adaptive methods

## Modern Enhancements

- **AdamW**: Adam with weight decay (better regularization)
- **NAdam**: Adam with Nesterov momentum
- **AMSGrad**: Fixes Adam convergence issues
- **Learning rate scheduling**: Decay η over time

## Code Architecture

```python
class Optimizer:
    def update(self, params, grads):
        # Each optimizer implements this differently
        pass

# Usage
optimizer = Adam(learning_rate=0.001)
updated_params = optimizer.update(params, grads)
```

## Visualization Insights

Check `/docs/optimizer_comparison.png`:
- **Loss curves**: RMSProp converges fastest, most stable
- **Convergence analysis**: RMSProp reaches low loss in fewest epochs
- **Final loss comparison**: RMSProp achieves lowest loss
- **Log scale**: Shows exponential convergence differences

## Next Steps

- **Learning rate scheduling**: Reduce η during training
- **Batch normalization**: Stabilize training further
- **Regularization**: Prevent overfitting
- **Second-order methods**: Newton's method, LBFGS