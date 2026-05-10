"""
Tiny Text Predictor Model

This module implements a simple next-token predictor using:
- Embeddings
- A single hidden layer
- Softmax output for categorical prediction
- Manual gradient descent updates

The model is designed for small language-modeling experiments and
already fits into the existing training/trainer pipeline.
"""

import numpy as np
from typing import List, Optional, Tuple
from .neural_networks.optimizers import get_optimizer

class TextPredictor:
    """Simple next-token predictor using embeddings and a feedforward classifier."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_size: int = 64,
                 learning_rate: float = 0.001, optimizer: str = 'adam'):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        # Embedding matrix and classifier weights
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W1 = np.random.randn(embed_dim, hidden_size) * np.sqrt(2.0 / embed_dim)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, vocab_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, vocab_size))

        optimizer_params = {'learning_rate': learning_rate}
        if optimizer == 'momentum':
            optimizer_params['momentum'] = 0.9
        elif optimizer == 'rmsprop':
            optimizer_params['beta'] = 0.9
            optimizer_params['epsilon'] = 1e-8
        elif optimizer == 'adam':
            optimizer_params['beta1'] = 0.9
            optimizer_params['beta2'] = 0.999
            optimizer_params['epsilon'] = 1e-8

        self.optimizer = get_optimizer(optimizer, **optimizer_params)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Apply softmax activation to logits."""
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute forward pass and return intermediate activations."""
        embeddings = self.embedding[X]  # shape: (batch_size, seq_len, embed_dim)
        pooled = np.mean(embeddings, axis=1)  # shape: (batch_size, embed_dim)

        z1 = np.dot(pooled, self.W1) + self.b1  # shape: (batch_size, hidden_size)
        a1 = self.relu(z1)  # shape: (batch_size, hidden_size)

        logits = np.dot(a1, self.W2) + self.b2  # shape: (batch_size, vocab_size)
        probs = self.softmax(logits)  # shape: (batch_size, vocab_size)

        return embeddings, pooled, z1, a1, logits, probs

    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """Compute categorical cross-entropy loss for a batch."""
        epsilon = 1e-15
        batch_size = probs.shape[0]
        correct_logprobs = -np.log(probs[np.arange(batch_size), y] + epsilon)
        return float(np.mean(correct_logprobs))

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform a single training step and update model parameters."""
        embeddings, pooled, z1, a1, logits, probs = self.forward(X)
        loss = self.compute_loss(probs, y)

        batch_size = X.shape[0]
        seq_len = X.shape[1]

        # Gradient of loss with respect to logits
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size

        dW2 = np.dot(a1.T, dlogits)
        db2 = np.sum(dlogits, axis=0, keepdims=True)

        dhidden = np.dot(dlogits, self.W2.T)
        dz1 = dhidden * self.relu_derivative(z1)

        dW1 = np.dot(pooled.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        dpooled = np.dot(dz1, self.W1.T)
        dembeddings = np.repeat(dpooled[:, np.newaxis, :] / seq_len, seq_len, axis=1)

        d_embedding = np.zeros_like(self.embedding)
        np.add.at(d_embedding, X, dembeddings)

        params = {
            'embedding': self.embedding,
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }
        grads = {
            'embedding': d_embedding,
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2,
        }

        updated = self.optimizer.update(params, grads)
        self.embedding = updated['embedding']
        self.W1 = updated['W1']
        self.b1 = updated['b1']
        self.W2 = updated['W2']
        self.b2 = updated['b2']

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the next-token index for each sequence in X."""
        _, _, _, _, _, probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def get_state(self) -> dict:
        """Return a serializable state dictionary for checkpointing."""
        return {
            'embedding': self.embedding.copy(),
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }

    def set_state(self, state: dict):
        """Restore parameters from a checkpoint state."""
        self.embedding = state['embedding'].copy()
        self.W1 = state['W1'].copy()
        self.b1 = state['b1'].copy()
        self.W2 = state['W2'].copy()
        self.b2 = state['b2'].copy()

    def generate(self, initial_tokens: List[int], seq_length: int,
                 max_new_tokens: int = 20, temperature: float = 1.0,
                 sample: bool = False) -> List[int]:
        """Generate a sequence of next-token predictions."""
        generated = list(initial_tokens)

        for _ in range(max_new_tokens):
            context = generated[-seq_length:]
            padded = np.array([context])
            _, _, _, _, _, probs = self.forward(padded)

            if temperature != 1.0:
                logits = np.log(probs + 1e-15) / max(temperature, 1e-8)
                probs = self.softmax(logits)

            if sample:
                next_token = int(np.random.choice(self.vocab_size, p=probs[0]))
            else:
                next_token = np.argmax(probs[0])
            generated.append(int(next_token))

        return generated
