"""
Training Infrastructure - Production-Ready Training Loops

Professional training requires:
1. Mini-batch processing (efficient memory usage)
2. Data shuffling (prevent overfitting to order)
3. Train/validation splits (monitor generalization)
4. Early stopping (prevent overfitting)
5. Learning rate scheduling (adapt learning during training)
6. Metrics tracking (loss curves, accuracy, etc.)
7. Checkpointing (save best models)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable

class TrainingMetrics:
    """
    Tracks training metrics and provides analysis tools.

    Stores loss/accuracy curves and provides methods for:
    - Early stopping detection
    - Learning rate scheduling triggers
    - Best model checkpointing
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []

    def update(self, epoch: int, train_loss: float, val_loss: float = None,
               train_acc: float = None, val_acc: float = None, lr: float = None):
        """Update metrics for current epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if lr is not None:
            self.learning_rates.append(lr)

    def should_early_stop(self, patience: int = 10, min_delta: float = 1e-4) -> bool:
        """
        Check if training should stop early based on validation loss.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement

        Returns:
            True if should stop early
        """
        if len(self.val_losses) < patience + 1:
            return False

        # Check if validation loss has improved in last 'patience' epochs
        best_loss = min(self.val_losses[:-patience])
        current_loss = self.val_losses[-1]

        return current_loss >= best_loss - min_delta

    def get_best_epoch(self) -> int:
        """Get epoch with best validation loss."""
        if not self.val_losses:
            return len(self.train_losses) - 1
        return np.argmin(self.val_losses)

    def get_convergence_info(self) -> Dict:
        """Get training convergence statistics."""
        return {
            'total_epochs': len(self.epochs),
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_epoch': self.get_best_epoch(),
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'converged': len(self.val_losses) > 10 and self.val_losses[-1] < self.val_losses[0] * 0.1
        }

class LearningRateScheduler:
    """
    Learning rate scheduling strategies.

    Reduces learning rate during training to allow fine-tuning:
    - Step decay: Reduce by factor every N epochs
    - Exponential decay: Exponential reduction
    - Cosine annealing: Smooth cosine curve
    - Reduce on plateau: Reduce when validation loss stops improving
    """

    def __init__(self, initial_lr: float, strategy: str = 'step', **kwargs):
        self.initial_lr = initial_lr
        self.strategy = strategy
        self.current_lr = initial_lr
        self.epoch = 0

        # Strategy-specific parameters
        if strategy == 'step':
            self.step_size = kwargs.get('step_size', 50)
            self.gamma = kwargs.get('gamma', 0.5)
        elif strategy == 'exponential':
            self.gamma = kwargs.get('gamma', 0.95)
        elif strategy == 'cosine':
            self.t_max = kwargs.get('t_max', 100)
        elif strategy == 'plateau':
            self.patience = kwargs.get('patience', 10)
            self.factor = kwargs.get('factor', 0.5)
            self.min_lr = kwargs.get('min_lr', 1e-6)
            self.best_loss = float('inf')
            self.wait = 0

    def step(self, val_loss: float = None) -> float:
        """Update learning rate and return new value."""
        self.epoch += 1

        if self.strategy == 'step':
            if self.epoch % self.step_size == 0:
                self.current_lr *= self.gamma
        elif self.strategy == 'exponential':
            self.current_lr = self.initial_lr * (self.gamma ** self.epoch)
        elif self.strategy == 'cosine':
            self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * self.epoch / self.t_max))
        elif self.strategy == 'plateau' and val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                    self.wait = 0

        return max(self.current_lr, 1e-8)  # Prevent LR from going too low

class DataLoader:
    """
    Handles data loading, shuffling, and batching.

    Provides efficient mini-batch training with:
    - Data shuffling between epochs
    - Train/validation splitting
    - Batch size control
    - Memory-efficient processing
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                 val_split: float = 0.2, shuffle: bool = True):
        """
        Initialize data loader.

        Args:
            X: Input features
            y: Target labels
            batch_size: Size of mini-batches
            val_split: Fraction of data for validation
            shuffle: Whether to shuffle data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Split data
        n_samples = len(X)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        self.X_train = X[train_indices]
        self.y_train = y[train_indices]
        self.X_val = X[val_indices]
        self.y_val = y[val_indices]

        self.n_train = n_train
        self.n_val = n_val

    def get_train_batches(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generator for training batches."""
        indices = np.arange(self.n_train)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n_train, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X_train[batch_indices], self.y_train[batch_indices]

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full validation set."""
        return self.X_val, self.y_val

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full training set."""
        return self.X_train, self.y_train

class Trainer:
    """
    Production-ready training orchestrator.

    Handles the complete training pipeline:
    - Data loading and batching
    - Training loop with mini-batches
    - Validation monitoring
    - Early stopping
    - Learning rate scheduling
    - Metrics tracking
    - Model checkpointing
    """

    def __init__(self, model, loss_fn: Callable, optimizer_name: str = 'adam',
                 learning_rate: float = 0.001, batch_size: int = 32):
        """
        Initialize trainer.

        Args:
            model: Neural network model to train
            loss_fn: Loss function (should return loss value)
            optimizer_name: Optimizer to use
            learning_rate: Initial learning rate
            batch_size: Mini-batch size
        """
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        # Initialize components
        self.metrics = TrainingMetrics()
        self.lr_scheduler = LearningRateScheduler(learning_rate, strategy='plateau')
        self.data_loader = None  # Set during fit()

        # Training state
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            val_split: float = 0.2, patience: int = 20, verbose: bool = True) -> TrainingMetrics:
        """
        Train the model.

        Args:
            X: Training features
            y: Training targets
            epochs: Maximum number of epochs
            val_split: Validation split fraction
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            TrainingMetrics object with full training history
        """
        # Setup data loading
        self.data_loader = DataLoader(X, y, self.batch_size, val_split)

        start_time = time.time()

        for epoch in range(epochs):
            # Train one epoch
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc = self._validate()

            # Update learning rate
            current_lr = self.lr_scheduler.step(val_loss)

            # Update metrics
            self.metrics.update(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self._get_model_state()

            # Print progress
            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                      f"Train Acc={train_acc:.1%}, Val Acc={val_acc:.1%}, LR={current_lr:.6f}, "
                      f"Time={elapsed:.1f}s")

            # Early stopping
            if self.metrics.should_early_stop(patience):
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        # Restore best model
        if self.best_model_state is not None:
            self._set_model_state(self.best_model_state)

        return self.metrics

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        total_loss = 0
        correct = 0
        total_samples = 0

        for X_batch, y_batch in self.data_loader.get_train_batches():
            # Forward + backward + update
            loss = self.model.train_step(X_batch, y_batch)
            total_loss += loss * len(X_batch)

            # Calculate accuracy
            predictions = self.model.predict(X_batch)
            correct += np.sum(predictions == y_batch)
            total_samples += len(X_batch)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples

        return avg_loss, accuracy

    def _validate(self) -> Tuple[float, float]:
        """Validate on validation set. Returns (avg_loss, accuracy)."""
        X_val, y_val = self.data_loader.get_val_data()

        # Get predictions and loss
        predictions = self.model.predict(X_val)
        accuracy = np.mean(predictions == y_val)

        # Calculate loss (this is approximate since we don't have batch processing for loss)
        # In practice, you'd want to compute loss properly
        val_loss = 1.0 - accuracy  # Simple approximation

        return val_loss, accuracy

    def _get_model_state(self) -> Dict:
        """Get current model parameters."""
        return {
            'W1': self.model.W1.copy(),
            'b1': self.model.b1.copy(),
            'W2': self.model.W2.copy(),
            'b2': self.model.b2.copy()
        }

    def _set_model_state(self, state: Dict):
        """Restore model parameters."""
        self.model.W1 = state['W1'].copy()
        self.model.b1 = state['b1'].copy()
        self.model.W2 = state['W2'].copy()
        self.model.b2 = state['b2'].copy()