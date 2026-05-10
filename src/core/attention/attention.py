"""
Attention mechanisms for transformer-style models.

This module includes the core building blocks used by transformer
architectures:
- Scaled dot-product attention
- Multi-head self-attention
- Positionwise feedforward layer
- Sinusoidal positional encoding
- Layer normalization
"""

import numpy as np
from typing import Optional, Tuple


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax implementation."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of softmax with respect to its inputs."""
    return y * (dy - np.sum(dy * y, axis=-1, keepdims=True))


class LayerNorm:
    """Layer normalization for transformer blocks."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))
        self.cache = {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        self.cache['x'] = x
        self.cache['x_norm'] = x_norm
        self.cache['mean'] = mean
        self.cache['variance'] = variance
        return self.gamma * x_norm + self.beta

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        mean = self.cache['mean']
        variance = self.cache['variance']
        std = np.sqrt(variance + self.eps)

        d_gamma = np.sum(d_out * x_norm, axis=(0, 1), keepdims=True)
        d_beta = np.sum(d_out, axis=(0, 1), keepdims=True)

        d_x_norm = d_out * self.gamma
        d_variance = np.sum(d_x_norm * (x - mean) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        d_mean = np.sum(d_x_norm * -1.0 / std, axis=-1, keepdims=True) + d_variance * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)

        d_x = d_x_norm / std + d_variance * 2.0 * (x - mean) / x.shape[-1] + d_mean / x.shape[-1]
        return d_x, d_gamma, d_beta


class PositionalEncoding:
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.max_len = max_len
        self.encoding = self._build_encoding()

    def _build_encoding(self) -> np.ndarray:
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        encoding = np.zeros((self.max_len, self.d_model))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)
        return encoding[np.newaxis, ...]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.encoding[:, :seq_len, :]


class ScaledDotProductAttention:
    """Scaled dot-product attention."""

    @staticmethod
    def forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        weights = softmax(scores, axis=-1)
        output = np.matmul(weights, V)
        return output, weights

    @staticmethod
    def backward(d_output: np.ndarray, weights: np.ndarray, Q: np.ndarray, K: np.ndarray,
                 V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dV = np.matmul(weights.transpose(0, 1, 3, 2), d_output)
        d_weights = np.matmul(d_output, V.transpose(0, 1, 3, 2))
        d_scores = softmax_backward(d_weights, weights)

        if mask is not None:
            d_scores = np.where(mask == 0, 0.0, d_scores)

        dQ = np.matmul(d_scores, K)
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
        return dQ, dK, dV


class MultiHeadAttention:
    """Multi-head self-attention layer."""

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.cache = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, x.shape[1], self.d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.cache['x'] = x
        self.cache['mask'] = mask

        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        attention_output, attention_weights = ScaledDotProductAttention.forward(Q_heads, K_heads, V_heads, mask)
        self.cache['attention_weights'] = attention_weights
        self.cache['attention_output'] = attention_output

        concat_attention = self._combine_heads(attention_output)
        self.cache['concat_attention'] = concat_attention

        output = np.matmul(concat_attention, self.W_o)
        return output, attention_weights

    def backward(self, d_output: np.ndarray) -> Tuple[np.ndarray, dict]:
        dW_o = np.dot(self.cache['concat_attention'].reshape(-1, self.d_model).T, d_output.reshape(-1, self.d_model))
        db_o = np.sum(d_output, axis=(0, 1), keepdims=True)

        d_concat = np.matmul(d_output, self.W_o.T)
        d_attention = self._split_heads(d_concat)

        dQ, dK, dV = ScaledDotProductAttention.backward(
            d_attention,
            self.cache['attention_weights'],
            self._split_heads(self.cache['Q']),
            self._split_heads(self.cache['K']),
            self._split_heads(self.cache['V']),
            self.cache['mask']
        )

        dQ = self._combine_heads(dQ)
        dK = self._combine_heads(dK)
        dV = self._combine_heads(dV)

        dW_q = np.dot(self.cache['x'].reshape(-1, self.d_model).T, dQ.reshape(-1, self.d_model))
        dW_k = np.dot(self.cache['x'].reshape(-1, self.d_model).T, dK.reshape(-1, self.d_model))
        dW_v = np.dot(self.cache['x'].reshape(-1, self.d_model).T, dV.reshape(-1, self.d_model))
        db_q = np.sum(dQ, axis=(0, 1), keepdims=True)
        db_k = np.sum(dK, axis=(0, 1), keepdims=True)
        db_v = np.sum(dV, axis=(0, 1), keepdims=True)

        d_input_q = np.dot(dQ.reshape(-1, self.d_model), self.W_q.T).reshape(self.cache['x'].shape)
        d_input_k = np.dot(dK.reshape(-1, self.d_model), self.W_k.T).reshape(self.cache['x'].shape)
        d_input_v = np.dot(dV.reshape(-1, self.d_model), self.W_v.T).reshape(self.cache['x'].shape)
        d_input = d_input_q + d_input_k + d_input_v

        grads = {
            'W_q': dW_q,
            'W_k': dW_k,
            'W_v': dW_v,
            'W_o': dW_o,
            'b_q': db_q,
            'b_k': db_k,
            'b_v': db_v,
            'b_o': db_o,
        }

        return d_input, grads


class PositionwiseFeedForward:
    """Feedforward network for transformer layers."""

    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros((1, d_model))
        self.cache = {}

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = self.relu(np.dot(x, self.W1) + self.b1)
        self.cache['x'] = x
        self.cache['hidden'] = hidden
        return np.dot(hidden, self.W2) + self.b2

    def backward(self, d_output: np.ndarray) -> Tuple[np.ndarray, dict]:
        batch_size, seq_len, d_model = d_output.shape
        flattened_hidden = self.cache['hidden'].reshape(-1, self.cache['hidden'].shape[-1])
        flattened_output = d_output.reshape(-1, d_model)

        d_hidden = np.dot(flattened_output, self.W2.T).reshape(batch_size, seq_len, -1)
        dW2 = np.dot(flattened_hidden.T, flattened_output)
        db2 = np.sum(flattened_output, axis=0, keepdims=True)

        d_relu = d_hidden * (self.cache['hidden'] > 0).astype(float)
        flattened_input = self.cache['x'].reshape(-1, self.cache['x'].shape[-1])
        flattened_relu = d_relu.reshape(-1, d_relu.shape[-1])

        dW1 = np.dot(flattened_input.T, flattened_relu)
        db1 = np.sum(flattened_relu, axis=0, keepdims=True)
        d_input = np.dot(flattened_relu, self.W1.T).reshape(batch_size, seq_len, -1)

        grads = {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2,
        }

        return d_input, grads
