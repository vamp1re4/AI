"""
Test the Perceptron on AND Gate Problem

AND gate truth table:
Input1 | Input2 | Output
   0   |   0    |   0
   0   |   1    |   0
   1   |   0    |   0
   1   |   1    |   1

The perceptron should learn to classify these patterns.
"""

import numpy as np
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.perceptron import Perceptron

def test_perceptron_and_gate():
    """Test perceptron learning AND gate logic."""

    # AND gate training data
    # Each row is [input1, input2], label is output
    X = np.array([
        [0, 0],  # 0 AND 0 = 0
        [0, 1],  # 0 AND 1 = 0
        [1, 0],  # 1 AND 0 = 0
        [1, 1]   # 1 AND 1 = 1
    ])

    y = np.array([0, 0, 0, 1])  # True labels

    print("AND Gate Training Data:")
    print("Input1 | Input2 | Output")
    for i, (x_i, y_i) in enumerate(zip(X, y)):
        print(f"   {x_i[0]}   |   {x_i[1]}    |   {y_i}")

    # Create perceptron with 2 inputs
    perceptron = Perceptron(n_features=2, learning_rate=0.1)

    print(f"\nInitial weights: {perceptron.weights}")
    print(f"Initial bias: {perceptron.bias}")

    # Train the perceptron
    print("\nTraining perceptron...")
    errors = perceptron.train(X, y, epochs=20, verbose=True)

    print(f"\nFinal weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias}")

    # Test predictions
    print("\nTesting predictions:")
    predictions = perceptron.predict(X)

    print("Input1 | Input2 | True | Predicted | Correct")
    all_correct = True
    for i, (x_i, y_true, y_pred) in enumerate(zip(X, y, predictions)):
        correct = "✓" if y_true == y_pred else "✗"
        if y_true != y_pred:
            all_correct = False
        print(f"   {x_i[0]}   |   {x_i[1]}    |  {y_true}   |    {y_pred}     |   {correct}")

    # Assert that perceptron learned correctly
    assert all_correct, "Perceptron failed to learn AND gate"
    assert np.array_equal(predictions, y), "Predictions don't match true labels"

    print("\n🎉 Success! Perceptron learned the AND gate perfectly!")

if __name__ == "__main__":
    test_perceptron_and_gate()