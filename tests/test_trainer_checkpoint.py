import os

import numpy as np
from src.core.text_model import TextPredictor
from src.core.training import Trainer


def test_trainer_saves_checkpoint(tmp_path):
    vocab_size = 10
    model = TextPredictor(vocab_size=vocab_size, embed_dim=8, hidden_size=16, learning_rate=0.01, optimizer='adam')

    X = np.random.randint(0, vocab_size, size=(20, 5))
    y = np.random.randint(0, vocab_size, size=(20,))

    trainer = Trainer(model, batch_size=4, checkpoint_dir=str(tmp_path))
    metrics = trainer.fit(X, y, epochs=3, val_split=0.2, patience=2, verbose=False)

    assert metrics is not None
    assert trainer.best_checkpoint_path is not None
    assert os.path.exists(trainer.best_checkpoint_path)

    loaded = np.load(trainer.best_checkpoint_path, allow_pickle=True)
    assert 'embedding' in loaded.files
    assert loaded['embedding'].shape == (vocab_size, 8)
