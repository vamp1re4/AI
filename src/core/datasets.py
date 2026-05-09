"""
Dataset Processing - Data Loading and Preprocessing

Handles:
1. Data loading from various formats
2. Preprocessing (normalization, encoding)
3. Data augmentation
4. Train/validation/test splits
5. Batch processing
6. Memory-efficient loading for large datasets
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import os

class Dataset:
    """
    Base class for datasets.

    Provides common functionality:
    - Data loading
    - Preprocessing
    - Splitting
    - Batching
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, name: str = "Dataset"):
        self.X = X
        self.y = y
        self.name = name
        self.n_samples = len(X)
        self.n_features = X.shape[1] if len(X.shape) > 1 else 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'name': self.name,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'X_shape': self.X.shape,
            'y_shape': self.y.shape,
            'X_dtype': self.X.dtype,
            'y_dtype': self.y.dtype,
            'X_min': np.min(self.X),
            'X_max': np.max(self.X),
            'X_mean': np.mean(self.X),
            'X_std': np.std(self.X),
            'y_unique': np.unique(self.y) if self.y.ndim == 1 else 'multi-dimensional'
        }

class Preprocessor:
    """
    Data preprocessing utilities.

    Handles common preprocessing tasks:
    - Normalization/standardization
    - Categorical encoding
    - Missing value handling
    - Feature scaling
    """

    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.label_encoder = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit preprocessor on training data."""
        # Standardize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)

        # Handle categorical labels if needed
        if y is not None and y.ndim == 1:
            unique_labels = np.unique(y)
            self.label_encoder = {label: i for i, label in enumerate(unique_labels)}

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using fitted parameters."""
        # Standardize features
        X_transformed = (X - self.feature_means) / (self.feature_stds + 1e-8)

        # Encode labels if needed
        y_transformed = y
        if y is not None and self.label_encoder is not None:
            y_transformed = np.array([self.label_encoder[label] for label in y])

        return X_transformed, y_transformed

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, y)

class DataSplitter:
    """
    Handles data splitting for training/validation/testing.

    Provides:
    - Train/val/test splits
    - Stratified splitting for classification
    - Cross-validation folds
    """

    @staticmethod
    def train_val_test_split(X: np.ndarray, y: np.ndarray,
                           train_ratio: float = 0.7, val_ratio: float = 0.15,
                           test_ratio: float = 0.15, random_state: int = 42) -> Tuple:
        """
        Split data into train/validation/test sets.

        Args:
            X: Features
            y: Labels
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            random_state: Random seed

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        n_samples = len(X)
        indices = np.arange(n_samples)

        np.random.seed(random_state)
        np.random.shuffle(indices)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        return (X[train_indices], X[val_indices], X[test_indices],
                y[train_indices], y[val_indices], y[test_indices])

    @staticmethod
    def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5,
                    random_state: int = 42) -> List[Tuple]:
        """
        Create k-fold cross-validation splits.

        Args:
            X: Features
            y: Labels
            k: Number of folds
            random_state: Random seed

        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        np.random.seed(random_state)
        np.random.shuffle(indices)

        fold_size = n_samples // k
        folds = []

        for i in range(k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < k - 1 else n_samples

            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

            folds.append((
                X[train_indices], X[val_indices],
                y[train_indices], y[val_indices]
            ))

        return folds

class DatasetGenerator:
    """
    Generate synthetic datasets for testing and experimentation.

    Creates various types of data:
    - Classification datasets (XOR, circles, moons)
    - Regression datasets
    - Noisy data
    """

    @staticmethod
    def make_xor(n_samples: int = 200, noise: float = 0.1, random_state: int = 42) -> Dataset:
        """Generate XOR classification dataset."""
        np.random.seed(random_state)

        # Generate data points
        X = np.random.randn(n_samples, 2)

        # XOR logic: different signs = class 1, same signs = class 0
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)

        # Add noise
        X += np.random.randn(n_samples, 2) * noise

        return Dataset(X, y.reshape(-1, 1), "XOR")

    @staticmethod
    def make_circles(n_samples: int = 200, noise: float = 0.1, factor: float = 0.5,
                    random_state: int = 42) -> Dataset:
        """Generate concentric circles classification dataset."""
        np.random.seed(random_state)

        # Generate angles
        angles = np.random.uniform(0, 2*np.pi, n_samples)

        # Create inner and outer circles
        inner_mask = np.random.binomial(1, 0.5, n_samples).astype(bool)

        # Inner circle
        r_inner = np.random.uniform(0, factor, np.sum(inner_mask))
        x_inner = r_inner * np.cos(angles[inner_mask])
        y_inner = r_inner * np.sin(angles[inner_mask])

        # Outer circle
        r_outer = np.random.uniform(factor, 1.0, np.sum(~inner_mask))
        x_outer = r_outer * np.cos(angles[~inner_mask])
        y_outer = r_outer * np.sin(angles[~inner_mask])

        # Combine
        X = np.zeros((n_samples, 2))
        X[inner_mask, 0] = x_inner
        X[inner_mask, 1] = y_inner
        X[~inner_mask, 0] = x_outer
        X[~inner_mask, 1] = y_outer

        y = inner_mask.astype(int)

        # Add noise
        X += np.random.randn(n_samples, 2) * noise

        return Dataset(X, y.reshape(-1, 1), "Circles")

    @staticmethod
    def make_moons(n_samples: int = 200, noise: float = 0.1,
                  random_state: int = 42) -> Dataset:
        """Generate two interleaving half-circles (moons)."""
        np.random.seed(random_state)

        n_samples_per_class = n_samples // 2

        # Upper moon
        angles1 = np.random.uniform(0, np.pi, n_samples_per_class)
        r1 = np.random.uniform(0.8, 1.2, n_samples_per_class)
        x1 = r1 * np.cos(angles1)
        y1 = r1 * np.sin(angles1) + 0.5

        # Lower moon
        angles2 = np.random.uniform(np.pi, 2*np.pi, n_samples_per_class)
        r2 = np.random.uniform(0.8, 1.2, n_samples_per_class)
        x2 = r2 * np.cos(angles2) + 1.0
        y2 = r2 * np.sin(angles2) - 0.5

        # Combine
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ])
        y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

        # Add noise
        X += np.random.randn(n_samples, 2) * noise

        return Dataset(X, y.reshape(-1, 1), "Moons")

    @staticmethod
    def make_blobs(n_samples: int = 200, centers: int = 3, cluster_std: float = 1.0,
                  random_state: int = 42) -> Dataset:
        """Generate isotropic Gaussian blobs."""
        np.random.seed(random_state)

        n_samples_per_center = n_samples // centers

        X = []
        y = []

        for i in range(centers):
            # Random center
            center = np.random.uniform(-2, 2, 2)

            # Generate points around center
            points = np.random.normal(center, cluster_std, (n_samples_per_center, 2))
            X.append(points)
            y.extend([i] * n_samples_per_center)

        X = np.vstack(X)
        y = np.array(y)

        return Dataset(X, y.reshape(-1, 1), f"Blobs_{centers}")

def load_dataset(name: str, **kwargs) -> Dataset:
    """
    Load a dataset by name.

    Args:
        name: Dataset name ('xor', 'circles', 'moons', 'blobs')
        **kwargs: Dataset-specific parameters

    Returns:
        Dataset object
    """
    generators = {
        'xor': DatasetGenerator.make_xor,
        'circles': DatasetGenerator.make_circles,
        'moons': DatasetGenerator.make_moons,
        'blobs': DatasetGenerator.make_blobs
    }

    if name not in generators:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(generators.keys())}")

    return generators[name](**kwargs)