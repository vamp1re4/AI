"""
Visualize ReLU vs Leaky ReLU on XOR Problem

ReLU (Rectified Linear Unit) is the most popular activation function because:
- Computationally efficient
- No vanishing gradients for positive inputs
- But can suffer from "dying ReLU" problem

Leaky ReLU fixes the dying ReLU problem by allowing small negative gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.mlp import MultiLayerPerceptron

def visualize_activations():
    """Compare ReLU and Leaky ReLU on XOR gate."""

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    activations = ['relu', 'leaky_relu']
    results = {}

    for activation in activations:
        print(f"Training with {activation.upper()}...")

        # Set seed for fair comparison
        np.random.seed(42)

        mlp = MultiLayerPerceptron(
            input_size=2,
            hidden_size=8,
            output_size=1,
            learning_rate=0.1,
            hidden_activation=activation,
            output_activation='sigmoid'
        )

        # Train
        losses = mlp.train(X, y, epochs=3000, batch_size=4, verbose=False)

        # Store results
        predictions = mlp.predict(X)
        accuracy = np.mean(predictions.flatten() == y.flatten())

        results[activation] = {
            'mlp': mlp,
            'losses': losses,
            'accuracy': accuracy,
            'predictions': predictions
        }

        print(".6f")
        print(".1%")

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ReLU vs Leaky ReLU on XOR Gate', fontsize=16)

    for i, activation in enumerate(activations):
        mlp = results[activation]['mlp']
        losses = results[activation]['losses']

        # Plot 1: Loss curve
        axes[i, 0].plot(losses, 'b-', linewidth=2)
        axes[i, 0].set_title(f'{activation.upper()} - Training Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss (BCE)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_yscale('log')

        # Plot 2: Decision boundary
        # Create grid for visualization
        x1_range = np.linspace(-0.5, 1.5, 100)
        x2_range = np.linspace(-0.5, 1.5, 100)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
        grid_points = np.c_[xx1.ravel(), xx2.ravel()]
        grid_predictions = mlp.predict(grid_points).reshape(xx1.shape)

        axes[i, 1].contourf(xx1, xx2, grid_predictions, alpha=0.3, cmap='RdBu', levels=[-0.5, 0.5, 1.5])
        axes[i, 1].scatter(X[y.flatten()==0][:, 0], X[y.flatten()==0][:, 1],
                          color='red', marker='o', label='XOR=0', s=100)
        axes[i, 1].scatter(X[y.flatten()==1][:, 0], X[y.flatten()==1][:, 1],
                          color='blue', marker='s', label='XOR=1', s=100)
        axes[i, 1].set_title(f'{activation.upper()} - Decision Boundary')
        axes[i, 1].set_xlabel('Input 1')
        axes[i, 1].set_ylabel('Input 2')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        # Plot 3: Activation distributions (hidden layer)
        _, a1, _, _ = mlp.forward(X)
        axes[i, 2].hist(a1.flatten(), bins=20, alpha=0.7, edgecolor='black')
        axes[i, 2].set_title(f'{activation.upper()} - Hidden Activations')
        axes[i, 2].set_xlabel('Activation Value')
        axes[i, 2].set_ylabel('Frequency')
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspaces/AI/docs/activation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    for activation in activations:
        acc = results[activation]['accuracy']
        final_loss = results[activation]['losses'][-1]
        status = "✅ Perfect" if acc == 1.0 else "⚠️ Good" if acc >= 0.75 else "❌ Poor"
        print("<12")

if __name__ == "__main__":
    visualize_activations()