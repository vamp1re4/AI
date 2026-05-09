"""
Demo script for training the tiny text predictor using real text sequences.

This script demonstrates:
- Building a tokenizer from sample text
- Creating next-token prediction sequences
- Training TextPredictor with Trainer
- Generating new text from a seed prompt
"""

import numpy as np
from src.core.text_processing import CharacterTokenizer, TextDataset, TextPreprocessor
from src.core.text_model import TextPredictor
from src.core.training import Trainer


def text_loss(model, X, y):
    """Loss wrapper for Trainer when using TextPredictor."""
    _, _, _, _, _, probs = model.forward(X)
    return model.compute_loss(probs, y)


def main():
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "This simple model learns to predict the next character.",
        "Language modeling is the foundation of modern text generation.",
        "A compact architecture can still learn useful patterns.",
        "Training on character sequences helps the model learn spelling and structure.",
        "Small models are useful for experiments and debugging.",
        "Next-token prediction is the core of autoregressive generation.",
        "Text generation becomes possible when the model learns sequence statistics."
    ]

    print("Building tokenizer...")
    tokenizer = CharacterTokenizer()
    tokenizer.fit(sample_texts)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    print("Creating next-token prediction dataset...")
    text_dataset = TextDataset(sample_texts, tokenizer)
    seq_length = 8
    X, y = text_dataset.create_next_token_sequences(seq_length=seq_length, step=1)

    print(f"Total examples: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("Initializing model...")
    model = TextPredictor(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=32,
        hidden_size=64,
        learning_rate=0.005,
        optimizer='adam'
    )

    trainer = Trainer(model, loss_fn=text_loss, batch_size=16)
    metrics = trainer.fit(X, y, epochs=120, val_split=0.2, patience=15, verbose=True)

    print("\nTraining completed")
    print(f"Epochs: {len(metrics.epochs)}")
    print(f"Final training loss: {metrics.train_losses[-1]:.5f}")
    print(f"Final validation loss: {metrics.val_losses[-1]:.5f}")

    # Generate text from a seed prompt
    seed = "the quick"
    seed_ids = tokenizer.encode(seed[-seq_length:])
    generated = model.generate(seed_ids, seq_length=seq_length, max_new_tokens=40, temperature=0.8)
    decoded = tokenizer.decode(generated)

    print("\nSeed:", seed)
    print("Generated:", decoded)


if __name__ == "__main__":
    main()
