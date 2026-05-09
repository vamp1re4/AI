"""
Visualize Optimizer Performance Comparison

Shows how different optimizers converge on the XOR problem:
- Training loss curves
- Convergence speed comparison
- Stability analysis

Key insights:
- Adam typically converges fastest
- RMSProp very stable with extreme probabilities
- Momentum helps with oscillations
- SGD reliable but slower
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.neural_networks.mlp import MultiLayerPerceptron

def visualize_optimizers():
    """Compare optimizer convergence on XOR gate."""

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # Optimizers with their best settings
    optimizers = {
        'SGD': {'optimizer': 'sgd', 'learning_rate': 0.1},
        'Momentum': {'optimizer': 'momentum', 'learning_rate': 0.1},
        'RMSProp': {'optimizer': 'rmsprop', 'learning_rate': 0.01},
        'Adam': {'optimizer': 'adam', 'learning_rate': 0.01}
    }

    results = {}

    for opt_name, opt_params in optimizers.items():
        print(f"Training with {opt_name}...")

        # Set seed for fair comparison
        np.random.seed(42)

        mlp = MultiLayerPerceptron(
            input_size=2,
            hidden_size=8,
            output_size=1,
            hidden_activation='leaky_relu',
            output_activation='sigmoid',
            **opt_params
        )

        # Train and collect losses
        losses = mlp.train(X, y, epochs=1000, batch_size=4, verbose=False)

        # Store results
        results[opt_name] = {
            'losses': losses,
            'mlp': mlp
        }

        print(".6f")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimizer Comparison on XOR Gate', fontsize=16)

    # Colors for different optimizers
    colors = {'SGD': 'blue', 'Momentum': 'green', 'RMSProp': 'red', 'Adam': 'purple'}

    # Plot 1: Loss curves (linear scale)
    ax1 = axes[0, 0]
    for opt_name, data in results.items():
        ax1.plot(data['losses'], color=colors[opt_name], linewidth=2, label=opt_name)
    ax1.set_title('Training Loss (Linear Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (BCE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss curves (log scale)
    ax2 = axes[0, 1]
    for opt_name, data in results.items():
        ax2.plot(data['losses'], color=colors[opt_name], linewidth=2, label=opt_name)
    ax2.set_title('Training Loss (Log Scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (BCE)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final loss comparison
    ax3 = axes[1, 0]
    opt_names = list(results.keys())
    final_losses = [results[opt]['losses'][-1] for opt in opt_names]
    bars = ax3.bar(opt_names, final_losses, color=[colors[opt] for opt in opt_names], alpha=0.7)
    ax3.set_title('Final Loss Comparison')
    ax3.set_ylabel('Final Loss (BCE)')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                '.6f', ha='center', va='bottom')

    # Plot 4: Convergence speed (epochs to reach loss < 0.1)
    ax4 = axes[1, 1]
    convergence_epochs = {}
    for opt_name, data in results.items():
        losses = data['losses']
        # Find first epoch where loss < 0.1
        for epoch, loss in enumerate(losses):
            if loss < 0.1:
                convergence_epochs[opt_name] = epoch
                break
        else:
            convergence_epochs[opt_name] = len(losses)  # Didn't converge

    conv_names = list(convergence_epochs.keys())
    conv_epochs = [convergence_epochs[opt] for opt in conv_names]
    bars = ax4.bar(conv_names, conv_epochs, color=[colors[opt] for opt in conv_names], alpha=0.7)
    ax4.set_title('Convergence Speed (Epochs to Loss < 0.1)')
    ax4.set_ylabel('Epochs')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, epochs in zip(bars, conv_epochs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{epochs}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('/workspaces/AI/docs/optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print detailed comparison
    print("\n" + "="*60)
    print("OPTIMIZER PERFORMANCE SUMMARY")
    print("="*60)
    print("<12")
    print("-" * 60)

    for opt_name in opt_names:
        final_loss = results[opt_name]['losses'][-1]
        conv_epoch = convergence_epochs[opt_name]
        print("<12")

    # Show which optimizer is best
    best_optimizer = min(results.keys(), key=lambda x: results[x]['losses'][-1])
    fastest_convergence = min(convergence_epochs.keys(), key=lambda x: convergence_epochs[x])

    print(f"\n🏆 Best final loss: {best_optimizer}")
    print(f"🏆 Fastest convergence: {fastest_convergence}")

if __name__ == "__main__":
    visualize_optimizers()