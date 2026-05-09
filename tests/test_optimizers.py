"""
Compare Optimization Algorithms on XOR Problem

This test demonstrates how different optimizers affect:
- Training speed and convergence
- Stability and oscillations
- Final performance
- Hyperparameter sensitivity

Optimizers tested:
- SGD: Basic gradient descent
- Momentum: Accelerated SGD
- RMSProp: Adaptive learning rates
- Adam: Momentum + RMSProp (best overall)
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.mlp import MultiLayerPerceptron

def compare_optimizers():
    """Compare different optimizers on XOR gate."""

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Optimizers to test with their best learning rates
    optimizers = {
        'sgd': {'learning_rate': 0.1},
        'momentum': {'learning_rate': 0.1, 'momentum': 0.9},
        'rmsprop': {'learning_rate': 0.01},  # RMSProp needs smaller LR
        'adam': {'learning_rate': 0.01}  # Adam works well with this LR
    }

    results = {}

    print("Comparing Optimization Algorithms on XOR Gate")
    print("=" * 60)

    for opt_name, opt_params in optimizers.items():
        print(f"\nTesting {opt_name.upper()} optimizer:")
        print("-" * 40)

        # Set seed for fair comparison
        np.random.seed(42)

        mlp = MultiLayerPerceptron(
            input_size=2,
            hidden_size=8,
            output_size=1,
            hidden_activation='leaky_relu',
            output_activation='sigmoid',
            optimizer=opt_name,
            learning_rate=opt_params['learning_rate']
        )

        print(f"Optimizer: {opt_name}")
        print(f"Parameters: {opt_params}")

        # Train
        losses = mlp.train(X, y, epochs=2000, batch_size=4, verbose=False)

        # Evaluate
        predictions = mlp.predict(X)
        probabilities = mlp.predict_proba(X)
        accuracy = np.mean(predictions.flatten() == y.flatten())

        print(".6f")
        print(".1%")
        print("Predictions:", predictions.flatten())
        print("Probabilities:", probabilities.flatten())

        # Check if learned correctly
        if accuracy == 1.0:
            print("✅ Perfect learning!")
        elif accuracy >= 0.75:
            print("⚠️ Good learning (3/4 correct)")
        else:
            print("❌ Poor learning")

        # Store results
        results[opt_name] = {
            'losses': losses,
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'params': opt_params
        }

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print("<12")
    print("-" * 60)

    for opt_name in optimizers.keys():
        loss = results[opt_name]['losses'][-1]
        acc = results[opt_name]['accuracy']
        lr = results[opt_name]['params']['learning_rate']
        status = "✅" if acc == 1.0 else "⚠️" if acc >= 0.75 else "❌"
        print("<12")

    return results

if __name__ == "__main__":
    results = compare_optimizers()