"""
Visualize Multi-Layer Perceptron Decision Boundary

Shows how a 2-layer network creates non-linear decision boundaries
that can solve XOR - something a single perceptron cannot do.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.mlp import MultiLayerPerceptron

def plot_decision_boundary_mlp(mlp, X, y, title="MLP Decision Boundary"):
    """
    Plot the decision boundary learned by the MLP.

    Since MLP has non-linear activations, the boundary is curved/complex.
    We'll evaluate predictions on a grid to visualize the boundary.
    """
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(X[y.flatten()==0][:, 0], X[y.flatten()==0][:, 1],
               color='red', marker='o', label='Class 0 (XOR=0)', s=100)
    plt.scatter(X[y.flatten()==1][:, 0], X[y.flatten()==1][:, 1],
               color='blue', marker='s', label='Class 1 (XOR=1)', s=100)

    # Create grid for decision boundary visualization
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    # Flatten grid for prediction
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]

    # Get predictions for all grid points
    predictions = mlp.predict(grid_points).reshape(xx1.shape)

    # Plot decision boundary (contour where prediction changes)
    plt.contourf(xx1, xx2, predictions, alpha=0.3, cmap='RdBu', levels=[-0.5, 0.5, 1.5])

    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xticks([0, 1])
    plt.yticks([0, 1])

def visualize_mlp_xor():
    """Visualize MLP learning XOR gate."""

    # Set random seed for reproducible results
    np.random.seed(42)

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Create and train MLP (same parameters as test)
    mlp = MultiLayerPerceptron(input_size=2, hidden_size=8, output_size=1, learning_rate=0.1)

    # Plot untrained network
    plot_decision_boundary_mlp(mlp, X, y, "Untrained MLP (Random Weights)")
    plt.savefig('/workspaces/AI/docs/mlp_untrained.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Train the network (same as test: 5000 epochs, batch_size=4)
    print("Training MLP...")
    losses = mlp.train(X, y, epochs=5000, batch_size=4, verbose=False)
    print(".6f")

    # Plot trained network
    plot_decision_boundary_mlp(mlp, X, y, "Trained MLP (Learned XOR)")
    plt.savefig('/workspaces/AI/docs/mlp_trained.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('MLP Training Loss on XOR Gate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see the decrease better
    plt.savefig('/workspaces/AI/docs/mlp_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Verify predictions
    predictions = mlp.predict(X)
    probabilities = mlp.predict_proba(X)
    print("Final predictions:", predictions.flatten())
    print("Probabilities:", probabilities.flatten())
    print("True labels:     ", y.flatten())
    print("All correct?", np.array_equal(predictions.flatten(), y.flatten()))

if __name__ == "__main__":
    visualize_mlp_xor()