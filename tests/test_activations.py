"""
Compare Different Activation Functions on XOR Problem

This test demonstrates how different activation functions affect:
- Training speed
- Final performance
- Gradient flow
- Convergence stability

Activation functions tested:
- Sigmoid: Classic, but vanishing gradients
- Tanh: Zero-centered, still vanishing gradients
- ReLU: Fast, no vanishing gradients for positive values
- Leaky ReLU: Fixes dying ReLU problem
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.mlp import MultiLayerPerceptron

def compare_activations():
    """Compare different activation functions on XOR gate."""

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Activation functions to test
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

    results = {}

    print("Comparing Activation Functions on XOR Gate")
    print("=" * 50)

    for activation in activations:
        print(f"\nTesting {activation.upper()} activation:")
        print("-" * 30)

        # Create network with this activation
        mlp = MultiLayerPerceptron(
            input_size=2,
            hidden_size=8,
            output_size=1,
            learning_rate=0.1,
            hidden_activation=activation,
            output_activation='sigmoid'  # Keep sigmoid for output (probabilities)
        )

        print(f"Hidden activation: {mlp.hidden_activation_name}")
        print(f"Output activation: {mlp.output_activation_name}")

        # Train
        losses = mlp.train(X, y, epochs=5000, batch_size=4, verbose=False)

        # Evaluate
        predictions = mlp.predict(X)
        probabilities = mlp.predict_proba(X)
        accuracy = np.mean(predictions.flatten() == y.flatten())

        print(".6f")
        print(".1%")
        print("Predictions:", predictions.flatten())
        print("Probabilities:", probabilities.flatten())

        # Store results
        results[activation] = {
            'final_loss': losses[-1],
            'accuracy': accuracy,
            'predictions': predictions.flatten(),
            'probabilities': probabilities.flatten(),
            'losses': losses
        }

        # Check if learned correctly
        if accuracy == 1.0:
            print("✅ Perfect learning!")
        elif accuracy >= 0.75:
            print("⚠️ Good learning (3/4 correct)")
        else:
            print("❌ Poor learning")

    # Summary comparison
    print("\n" + "=" * 50)
    print("SUMMARY COMPARISON")
    print("=" * 50)
    print("<12")
    print("-" * 50)

    for activation in activations:
        loss = results[activation]['final_loss']
        acc = results[activation]['accuracy']
        status = "✅" if acc == 1.0 else "⚠️" if acc >= 0.75 else "❌"
        print("<12")

    return results

if __name__ == "__main__":
    results = compare_activations()