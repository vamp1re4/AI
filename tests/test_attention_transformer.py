"""
Tests for attention and transformer modules.
"""

import numpy as np
from src.core.attention.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    LayerNorm
)
from src.core.transformers.transformer import TransformerPredictor, TransformerEncoderLayer


def test_scaled_dot_product_attention_shape_and_weights():
    batch_size, num_heads, seq_len, d_head = 2, 4, 5, 8
    Q = np.random.randn(batch_size, num_heads, seq_len, d_head)
    K = np.random.randn(batch_size, num_heads, seq_len, d_head)
    V = np.random.randn(batch_size, num_heads, seq_len, d_head)

    output, weights = ScaledDotProductAttention.forward(Q, K, V)

    assert output.shape == (batch_size, num_heads, seq_len, d_head)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-6)


def test_multi_head_attention_output_shape():
    batch_size, seq_len, d_model = 2, 6, 32
    num_heads = 4
    x = np.random.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output, weights = mha.forward(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_positional_encoding_adds_positions():
    d_model = 16
    seq_len = 10
    encoder = PositionalEncoding(d_model, max_len=50)
    x = np.zeros((1, seq_len, d_model))

    encoded = encoder(x)
    assert encoded.shape == x.shape
    assert not np.allclose(encoded[0, 0], encoded[0, 1])


def test_layer_norm_preserves_shape():
    x = np.random.randn(3, 4, 8)
    layer_norm = LayerNorm(8)
    output = layer_norm(x)

    assert output.shape == x.shape
    assert np.allclose(np.mean(output, axis=-1), np.zeros((3, 4)), atol=1e-5)
    assert np.allclose(np.std(output, axis=-1), np.ones((3, 4)), atol=1e-5)


def test_transformer_predictor_train_and_predict():
    vocab_size = 20
    seq_length = 6
    model = TransformerPredictor(vocab_size=vocab_size, d_model=32, num_heads=4, d_ff=64, num_layers=1, max_seq_len=seq_length)

    X = np.random.randint(0, vocab_size, size=(12, seq_length))
    y = np.random.randint(0, vocab_size, size=(12,))

    initial_loss = model.train_step(X, y)
    predictions = model.predict(X)

    assert initial_loss >= 0
    assert predictions.shape == (12,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < vocab_size)


def test_transformer_encoder_layer_shape():
    batch_size, seq_len, d_model = 3, 5, 32
    layer = TransformerEncoderLayer(d_model=d_model, num_heads=4, d_ff=64)
    x = np.random.randn(batch_size, seq_len, d_model)
    output = layer.forward(x)

    assert output.shape == x.shape
