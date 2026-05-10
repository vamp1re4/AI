"""
Tests for training transformer-based text models with the generic Trainer.
"""

import numpy as np
from src.core.text_processing import CharacterTokenizer, TextDataset
from src.core.transformers.transformer import TransformerPredictor
from src.core.training import Trainer


def test_trainer_can_train_transformer_predictor():
    texts = ["hello world"]
    tokenizer = CharacterTokenizer()
    tokenizer.fit(texts)

    dataset = TextDataset(texts, tokenizer)
    X, y = dataset.create_next_token_sequences(seq_length=4)

    model = TransformerPredictor(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=16,
        num_heads=4,
        d_ff=32,
        num_layers=1,
        max_seq_len=4,
        learning_rate=0.01,
        optimizer='adam'
    )

    trainer = Trainer(
        model,
        loss_fn=lambda m, X_batch, y_batch: m.compute_loss(m.forward(X_batch)[-1], y_batch),
        batch_size=2
    )

    metrics = trainer.fit(X, y, epochs=3, val_split=0.2, patience=2, verbose=False)

    assert len(metrics.epochs) > 0
    assert np.all(np.array(metrics.train_losses) >= 0)
    assert metrics.val_losses[-1] >= 0
    predictions = model.predict(X[:2])
    assert predictions.shape == (2,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < tokenizer.get_vocab_size())
