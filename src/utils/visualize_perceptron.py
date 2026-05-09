"""
Visualize Perceptron Learning

This script shows how the perceptron's decision boundary evolves during training.
We'll plot the AND gate points and the separating line that the perceptron learns.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.perceptron import Perceptron

def plot_decision_boundary(perceptron, X, y, title="Perceptron Decision Boundary"):
    """
    Plot the decision boundary learned by the perceptron.

    The decision boundary is where: weights · x + bias = 0
    So: x2 = (-bias - w1*x1) / w2
    """
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', marker='o', label='Class 0 (False)', s=100)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', marker='s', label='Class 1 (True)', s=100)

    # Plot decision boundary if perceptron is trained
    if hasattr(perceptron, 'weights'):
        w1, w2 = perceptron.weights
        b = perceptron.bias

        # Create x1 values for plotting
        x1_range = np.linspace(-0.5, 1.5, 100)

        # Calculate corresponding x2 values: w1*x1 + w2*x2 + b = 0 => x2 = (-b - w1*x1)/w2
        if w2 != 0:
            x2_boundary = (-b - w1 * x1_range) / w2
            plt.plot(x1_range, x2_boundary, 'g-', linewidth=2, label='Decision Boundary')
        else:
            # Vertical line if w2 = 0
            plt.axvline(x=-b/w1 if w1 != 0 else 0, color='g', linewidth=2, label='Decision Boundary')

    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xticks([0, 1])
    plt.yticks([0, 1])

def visualize_perceptron_learning():
    """Visualize how perceptron learns AND gate over time."""

    # AND gate data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    # Create perceptron
    perceptron = Perceptron(n_features=2, learning_rate=0.1)

    # Plot initial state
    plot_decision_boundary(perceptron, X, y, "Initial Random Weights")
    plt.savefig('/workspaces/AI/docs/perceptron_initial.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Train for a few epochs and show progress
    epochs_to_show = [0, 5, 10, 20]
    for epoch in epochs_to_show[1:]:
        perceptron.train(X, y, epochs=epoch, verbose=False)
        plot_decision_boundary(perceptron, X, y, f"After {epoch} Training Epochs")
        plt.savefig(f'/workspaces/AI/docs/perceptron_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Final result
    print("Final weights:", perceptron.weights)
    print("Final bias:", perceptron.bias)

    # Verify predictions
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
    print("True labels:", y)
    print("All correct?", np.array_equal(predictions, y))

if __name__ == "__main__":
    visualize_perceptron_learning()