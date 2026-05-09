"""
Tests for the tiny text predictor model and next-token sequence utilities.
"""

import numpy as np
from src.core.text_processing import CharacterTokenizer, TextDataset, TextPreprocessor
from src.core.text_model import TextPredictor
from src.core.training import Trainer

class TestTextPreprocessorNextTokenSequences:
    def test_create_next_token_sequences(self):
        text = "hello world"
        tokenizer = CharacterTokenizer()
        tokenizer.fit([text])

        preprocessor = TextPreprocessor()
        X, y = preprocessor.create_next_token_sequences(text, seq_length=3, tokenizer=tokenizer)

        assert X.shape[1] == 3
        assert y.shape[0] == X.shape[0]
        assert all(isinstance(item, np.ndarray) or isinstance(item, np.generic) for item in np.atleast_1d(y))

    def test_text_dataset_next_token_sequences(self):
        texts = ["hello world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        dataset = TextDataset(texts, tokenizer)
        X, y = dataset.create_next_token_sequences(seq_length=4)

        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 4

class TestTextPredictor:
    def test_predictor_forward_and_predict(self):
        texts = ["hello world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        seq_length = 5
        vocab_size = tokenizer.get_vocab_size()
        model = TextPredictor(vocab_size=vocab_size, embed_dim=16, hidden_size=32, learning_rate=0.01, optimizer='sgd')

        X = np.array([tokenizer.encode("hello")[:seq_length]])
        y = np.array([tokenizer.encode("hello world")[seq_length]])

        loss = model.train_step(X, y)
        predictions = model.predict(X)

        assert loss > 0
        assert predictions.shape == (1,)
        assert predictions.dtype == np.int64 or predictions.dtype == np.int32

    def test_generate_returns_expected_length(self):
        texts = ["hello world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        model = TextPredictor(vocab_size=tokenizer.get_vocab_size(), embed_dim=16, hidden_size=32)
        initial_tokens = tokenizer.encode("hello")

        generated = model.generate(initial_tokens=initial_tokens, seq_length=4, max_new_tokens=5)

        assert len(generated) == len(initial_tokens) + 5
        assert all(isinstance(token, int) for token in generated)

    def test_trainer_trains_text_predictor(self):
        texts = ["hello world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        dataset = TextDataset(texts, tokenizer)
        X, y = dataset.create_next_token_sequences(seq_length=4)

        model = TextPredictor(vocab_size=tokenizer.get_vocab_size(), embed_dim=8, hidden_size=16, learning_rate=0.01)
        trainer = Trainer(
            model,
            loss_fn=lambda model, X, y: model.compute_loss(model.forward(X)[5], y),
            batch_size=2
        )

        metrics = trainer.fit(X, y, epochs=5, val_split=0.2, patience=2, verbose=False)

        assert len(metrics.epochs) > 0
        assert hasattr(model, 'predict')
        assert metrics.train_losses[-1] >= 0.0
