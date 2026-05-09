"""
Test Multi-Layer Perceptron on XOR Gate

XOR is impossible for a single perceptron but easy for multi-layer networks.
This demonstrates the power of hidden layers and non-linear transformations.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.mlp import MultiLayerPerceptron

def test_mlp_xor_gate():
    """Test MLP learning XOR gate (non-linearly separable problem)."""

    # Set same seed as visualization for comparison
    np.random.seed(42)

    # XOR gate training data
    # XOR: different inputs -> 1, same inputs -> 0
    X = np.array([
        [0, 0],  # 0 XOR 0 = 0
        [0, 1],  # 0 XOR 1 = 1
        [1, 0],  # 1 XOR 0 = 1
        [1, 1]   # 1 XOR 1 = 0
    ], dtype=float)

    y = np.array([
        [0],  # Output as column vector for matrix operations
        [1],
        [1],
        [0]
    ], dtype=float)

    print("XOR Gate Training Data:")
    print("Input1 | Input2 | Output")
    for i, (x_i, y_i) in enumerate(zip(X, y.flatten())):
        print(f"   {int(x_i[0])}   |   {int(x_i[1])}    |   {int(y_i)}")

    # Create MLP: 2 inputs -> 8 hidden neurons -> 1 output
    mlp = MultiLayerPerceptron(input_size=2, hidden_size=8, output_size=1, learning_rate=0.1)

    print("\nNetwork Architecture:")
    print(f"Input layer: {mlp.W1.shape[0]} neurons")
    print(f"Hidden layer: {mlp.W1.shape[1]} neurons")
    print(f"Output layer: {mlp.W2.shape[1]} neurons")
    print(f"Total parameters: {mlp.W1.size + mlp.b1.size + mlp.W2.size + mlp.b2.size}")

    # Train the network
    print("\nTraining MLP on XOR gate...")
    losses = mlp.train(X, y, epochs=5000, batch_size=4, verbose=True)

    print(".6f")

    # Test predictions
    print("\nTesting predictions:")
    predictions = mlp.predict(X)
    probabilities = mlp.predict_proba(X)

    print("Input1 | Input2 | True | Predicted | Probability | Correct")
    all_correct = True
    for i, (x_i, y_true, y_pred, prob) in enumerate(zip(X, y.flatten(), predictions.flatten(), probabilities.flatten())):
        correct = "✓" if y_true == y_pred else "✗"
        if y_true != y_pred:
            all_correct = False
        print(f"   {int(x_i[0])}   |   {int(x_i[1])}    |  {int(y_true)}   |    {int(y_pred)}     |    {prob:.3f}    |   {correct}")

    # Assert success
    assert all_correct, "MLP failed to learn XOR gate"
    assert np.array_equal(predictions.flatten(), y.flatten()), "Predictions don't match true labels"

    print("\n🎉 Success! MLP learned XOR gate perfectly!")
    print("This proves multi-layer networks can solve non-linear problems!")

    return losses

if __name__ == "__main__":
    losses = test_mlp_xor_gate()