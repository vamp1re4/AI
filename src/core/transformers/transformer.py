"""
Minimal transformer implementation for next-token prediction.

This module provides a simple transformer encoder block and
an autoregressive predictor that uses self-attention.
"""

import numpy as np
from typing import Optional, Tuple
from src.core.attention.attention import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    LayerNorm,
    PositionalEncoding
)
from src.core.neural_networks.optimizers import get_optimizer


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create a causal mask for autoregressive attention."""
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
    return mask[np.newaxis, np.newaxis, :, :]


class TransformerEncoderLayer:
    """Single transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.layer_norm2 = LayerNorm(d_model)
        self.cache = {}

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        attention_output, _ = self.self_attention.forward(x, mask)
        residual1 = x + attention_output
        normalized1 = self.layer_norm1(residual1)
        feedforward_output = self.feed_forward.forward(normalized1)
        residual2 = normalized1 + feedforward_output
        output = self.layer_norm2(residual2)

        self.cache['x'] = x
        self.cache['residual1'] = residual1
        self.cache['normalized1'] = normalized1
        self.cache['residual2'] = residual2

        return output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        d_residual2, d_gamma2, d_beta2 = self.layer_norm2.backward(d_output)
        d_normalized1 = d_residual2
        d_feedforward_input, ff_grads = self.feed_forward.backward(d_normalized1)

        d_normalized1 += d_feedforward_input
        d_residual1, d_gamma1, d_beta1 = self.layer_norm1.backward(d_normalized1)

        d_x = d_residual1
        d_attention_input, attn_grads = self.self_attention.backward(d_residual1)
        d_x += d_attention_input

        self.cache['gradients'] = {
            'layer_norm1_gamma': d_gamma1,
            'layer_norm1_beta': d_beta1,
            'layer_norm2_gamma': d_gamma2,
            'layer_norm2_beta': d_beta2,
            **ff_grads,
            **attn_grads
        }

        return d_x

    def get_parameters(self) -> dict:
        return {
            'W_q': self.self_attention.W_q,
            'W_k': self.self_attention.W_k,
            'W_v': self.self_attention.W_v,
            'W_o': self.self_attention.W_o,
            'W1': self.feed_forward.W1,
            'b1': self.feed_forward.b1,
            'W2': self.feed_forward.W2,
            'b2': self.feed_forward.b2,
            'layer_norm1_gamma': self.layer_norm1.gamma,
            'layer_norm1_beta': self.layer_norm1.beta,
            'layer_norm2_gamma': self.layer_norm2.gamma,
            'layer_norm2_beta': self.layer_norm2.beta,
        }


class TransformerEncoder:
    """Stack of transformer encoder layers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        self.layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        grad = d_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_parameters(self) -> dict:
        params = {}
        for idx, layer in enumerate(self.layers):
            params.update({f'layer_{idx}_{k}': v for k, v in layer.get_parameters().items()})
        return params

    def get_gradients(self) -> dict:
        grads = {}
        for idx, layer in enumerate(self.layers):
            layer_grads = layer.cache.get('gradients', {})
            for key, value in layer_grads.items():
                grads[f'layer_{idx}_{key}'] = value
        return grads


class TransformerPredictor:
    """Transformer-based next-token predictor."""

    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4,
                 d_ff: int = 128, num_layers: int = 2, max_seq_len: int = 32,
                 learning_rate: float = 0.001, optimizer: str = 'adam', optimizer_params: Optional[dict] = None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
        self.output_layer = np.random.randn(d_model, vocab_size) * np.sqrt(1.0 / d_model)
        self.output_bias = np.zeros((1, vocab_size))

        opt_params = {'learning_rate': learning_rate}
        if optimizer_params is not None:
            opt_params.update(optimizer_params)

        if optimizer == 'momentum' and 'momentum' not in opt_params:
            opt_params['momentum'] = 0.9
        if optimizer == 'rmsprop':
            opt_params.setdefault('beta', 0.9)
            opt_params.setdefault('epsilon', 1e-8)
        if optimizer == 'adam':
            opt_params.setdefault('beta1', 0.9)
            opt_params.setdefault('beta2', 0.999)
            opt_params.setdefault('epsilon', 1e-8)

        self.optimizer = get_optimizer(optimizer, **opt_params)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, seq_len = X.shape
        embeddings = self.embedding[X]
        embeddings = self.positional_encoding(embeddings)

        if mask is None:
            mask = create_causal_mask(seq_len)

        encoded = self.encoder.forward(embeddings, mask)
        logits = np.dot(encoded[:, -1, :], self.output_layer) + self.output_bias
        probs = self.softmax(logits)
        return encoded, logits, probs

    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        epsilon = 1e-15
        batch_size = probs.shape[0]
        correct_logprobs = -np.log(probs[np.arange(batch_size), y] + epsilon)
        return float(np.mean(correct_logprobs))

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        encoded, logits, probs = self.forward(X)
        loss = self.compute_loss(probs, y)

        batch_size = X.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size

        last_hidden = encoded[:, -1, :]
        dW_out = np.dot(last_hidden.T, dlogits)
        dbias = np.sum(dlogits, axis=0, keepdims=True)

        d_last_hidden = np.dot(dlogits, self.output_layer.T)
        d_encoded = np.zeros_like(encoded)
        d_encoded[:, -1, :] = d_last_hidden

        d_embeddings = self.encoder.backward(d_encoded)

        d_embedding_matrix = np.zeros_like(self.embedding)
        np.add.at(d_embedding_matrix, X.reshape(-1), d_embeddings.reshape(-1, self.d_model))

        params = {
            'embedding': self.embedding,
            'output_layer': self.output_layer,
            'output_bias': self.output_bias,
        }
        grads = {
            'embedding': d_embedding_matrix,
            'output_layer': dW_out,
            'output_bias': dbias,
        }

        encoder_params = self.encoder.get_parameters()
        encoder_grads = self.encoder.get_gradients()
        params.update(encoder_params)
        grads.update(encoder_grads)

        updated = self.optimizer.update(params, grads)
        self.embedding = updated['embedding']
        self.output_layer = updated['output_layer']
        self.output_bias = updated['output_bias']

        for key in encoder_params.keys():
            _, layer_idx, param_name = key.split('_', 2)
            layer = self.encoder.layers[int(layer_idx)]

            if param_name in {'W_q', 'W_k', 'W_v', 'W_o'}:
                setattr(layer.self_attention, param_name, updated[key])
            elif param_name in {'W1', 'b1', 'W2', 'b2'}:
                setattr(layer.feed_forward, param_name, updated[key])
            elif param_name == 'layer_norm1_gamma':
                layer.layer_norm1.gamma = updated[key]
            elif param_name == 'layer_norm1_beta':
                layer.layer_norm1.beta = updated[key]
            elif param_name == 'layer_norm2_gamma':
                layer.layer_norm2.gamma = updated[key]
            elif param_name == 'layer_norm2_beta':
                layer.layer_norm2.beta = updated[key]

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def generate(self, initial_tokens: np.ndarray, max_new_tokens: int = 20,
                 temperature: float = 1.0, sample: bool = False) -> np.ndarray:
        generated = list(initial_tokens)
        for _ in range(max_new_tokens):
            context = np.array([generated[-self.max_seq_len:]])
            _, _, probs = self.forward(context)

            if temperature != 1.0:
                logits = np.log(probs + 1e-15) / max(temperature, 1e-8)
                probs = self.softmax(logits)

            if sample:
                next_token = int(np.random.choice(self.vocab_size, p=probs[0]))
            else:
                next_token = int(np.argmax(probs[0]))

            generated.append(next_token)
        return np.array(generated)

    def get_state(self) -> dict:
        state = {
            'embedding': self.embedding.copy(),
            'output_layer': self.output_layer.copy(),
            'output_bias': self.output_bias.copy()
        }
        state.update(self.encoder.get_parameters())
        return state

    def set_state(self, state: dict):
        self.embedding = state['embedding'].copy()
        self.output_layer = state['output_layer'].copy()
        self.output_bias = state['output_bias'].copy()

        for key, value in state.items():
            if key in {'embedding', 'output_layer', 'output_bias'}:
                continue
            _, layer_idx, param_name = key.split('_', 2)
            layer = self.encoder.layers[int(layer_idx)]
            if param_name in {'W_q', 'W_k', 'W_v', 'W_o'}:
                setattr(layer.self_attention, param_name, value.copy())
            elif param_name in {'W1', 'b1', 'W2', 'b2'}:
                setattr(layer.feed_forward, param_name, value.copy())
            elif param_name == 'layer_norm1_gamma':
                layer.layer_norm1.gamma = value.copy()
            elif param_name == 'layer_norm1_beta':
                layer.layer_norm1.beta = value.copy()
            elif param_name == 'layer_norm2_gamma':
                layer.layer_norm2.gamma = value.copy()
            elif param_name == 'layer_norm2_beta':
                layer.layer_norm2.beta = value.copy()
