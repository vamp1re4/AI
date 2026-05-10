"""
Mini language model wrapper for transformer-based next-token prediction.

This module provides a reusable API for training, saving, loading, and
generating text with a small transformer-based language model.
"""

import os
from typing import List, Optional

import numpy as np

from .text_processing import TextDataset, Tokenizer
from .training import Trainer
from .transformers.transformer import TransformerPredictor


class MiniLanguageModel:
    """Mini language model wrapper around TransformerPredictor."""

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 d_ff: int = 128,
                 num_layers: int = 2,
                 max_seq_len: int = 32,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 optimizer_params: Optional[dict] = None):
        self.model = TransformerPredictor(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_params=optimizer_params
        )

    def train(self,
              texts: List[str],
              tokenizer: Tokenizer,
              seq_length: int,
              epochs: int = 20,
              batch_size: int = 32,
              val_split: float = 0.2,
              patience: int = 10,
              verbose: bool = True,
              checkpoint_dir: Optional[str] = None):
        """Train the mini language model on raw text data."""
        dataset = TextDataset(texts, tokenizer)
        X, y = dataset.create_next_token_sequences(seq_length)

        if X.size == 0:
            raise ValueError('Not enough data to create training sequences.')

        trainer = Trainer(self.model, batch_size=batch_size, checkpoint_dir=checkpoint_dir)
        return trainer.fit(X, y, epochs=epochs, val_split=val_split,
                           patience=patience, verbose=verbose)

    def generate(self,
                 initial_tokens: List[int],
                 max_new_tokens: int = 20,
                 temperature: float = 1.0,
                 sample: bool = False) -> List[int]:
        """Generate text from an initial token sequence."""
        initial_array = np.array(initial_tokens, dtype=int)
        return self.model.generate(initial_array, max_new_tokens=max_new_tokens,
                                   temperature=temperature, sample=sample)

    def save(self, filepath: str) -> None:
        """Save model weights to disk."""
        state = self.model.get_state()
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(filepath, **state)

    def load(self, filepath: str) -> None:
        """Load model weights from disk."""
        loaded = np.load(filepath, allow_pickle=True)
        state = {key: loaded[key] for key in loaded.files}
        self.model.set_state(state)
