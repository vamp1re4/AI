"""
Tests for Dataset Processing and Text Processing modules.
"""

import numpy as np
import pytest
from src.core.datasets import (
    Dataset, Preprocessor, DataSplitter, DatasetGenerator,
    load_dataset
)
from src.core.text_processing import (
    CharacterTokenizer, WordTokenizer, TextPreprocessor,
    TextDataset, load_text_file, save_text_file
)

class TestDataset:
    """Test Dataset class."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])

        dataset = Dataset(X, y, "test")

        assert len(dataset) == 3
        assert dataset.n_samples == 3
        assert dataset.n_features == 2
        assert dataset.name == "test"

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        dataset = Dataset(X, y)

        x_item, y_item = dataset[0]
        assert np.array_equal(x_item, [1, 2])
        assert y_item == 0

    def test_dataset_stats(self):
        """Test dataset statistics."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        dataset = Dataset(X, y, "stats_test")
        stats = dataset.get_stats()

        assert stats['name'] == "stats_test"
        assert stats['n_samples'] == 2
        assert stats['n_features'] == 2
        assert stats['X_min'] == 1.0
        assert stats['X_max'] == 4.0

class TestPreprocessor:
    """Test Preprocessor class."""

    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        preprocessor = Preprocessor()
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)

        # Check standardization
        assert abs(np.mean(X_transformed)) < 1e-10  # Should be approximately 0
        assert abs(np.std(X_transformed) - 1.0) < 1e-6  # Should be approximately 1 (relaxed tolerance)

        # Check labels unchanged for this case
        assert np.array_equal(y_transformed, y)

    def test_preprocessor_transform_only(self):
        """Test preprocessor transform without fit."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        preprocessor = Preprocessor()
        # Should raise error if not fitted
        with pytest.raises(TypeError):
            preprocessor.transform(X)

class TestDataSplitter:
    """Test DataSplitter class."""

    def test_train_val_test_split(self):
        """Test train/validation/test splitting."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)

        X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.train_val_test_split(
            X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        assert len(X_train) == 6
        assert len(X_val) == 2
        assert len(X_test) == 2

        # Check no overlap in indices
        train_indices = set()
        for x in X_train:
            for i, orig_x in enumerate(X):
                if np.array_equal(x, orig_x):
                    train_indices.add(i)

        val_indices = set()
        for x in X_val:
            for i, orig_x in enumerate(X):
                if np.array_equal(x, orig_x):
                    val_indices.add(i)

        test_indices = set()
        for x in X_test:
            for i, orig_x in enumerate(X):
                if np.array_equal(x, orig_x):
                    test_indices.add(i)

        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0

    def test_k_fold_split(self):
        """Test k-fold cross-validation splitting."""
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)

        folds = DataSplitter.k_fold_split(X, y, k=5)

        assert len(folds) == 5

        for X_train, X_val, y_train, y_val in folds:
            assert len(X_val) == 2  # 10 samples / 5 folds = 2 per fold
            assert len(X_train) == 8

class TestDatasetGenerator:
    """Test DatasetGenerator class."""

    def test_make_xor(self):
        """Test XOR dataset generation."""
        dataset = DatasetGenerator.make_xor(n_samples=100)

        assert len(dataset) == 100
        assert dataset.n_features == 2
        assert dataset.name == "XOR"

        # Check that labels are binary (0 or 1)
        y = dataset.y.flatten()
        assert set(y) <= {0, 1}  # All labels should be 0 or 1

    def test_make_circles(self):
        """Test circles dataset generation."""
        dataset = DatasetGenerator.make_circles(n_samples=100)

        assert len(dataset) == 100
        assert dataset.n_features == 2
        assert dataset.name == "Circles"

    def test_make_moons(self):
        """Test moons dataset generation."""
        dataset = DatasetGenerator.make_moons(n_samples=100)

        assert len(dataset) == 100
        assert dataset.n_features == 2
        assert dataset.name == "Moons"

    def test_make_blobs(self):
        """Test blobs dataset generation."""
        dataset = DatasetGenerator.make_blobs(n_samples=99, centers=3)

        assert len(dataset) == 99
        assert dataset.n_features == 2
        assert dataset.name == "Blobs_3"

class TestLoadDataset:
    """Test load_dataset function."""

    def test_load_xor(self):
        """Test loading XOR dataset."""
        dataset = load_dataset('xor', n_samples=50)
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 50
        assert dataset.name == "XOR"

    def test_load_unknown_dataset(self):
        """Test loading unknown dataset raises error."""
        with pytest.raises(ValueError):
            load_dataset('unknown')

class TestCharacterTokenizer:
    """Test CharacterTokenizer class."""

    def test_fit_and_vocab(self):
        """Test fitting and vocabulary building."""
        texts = ["hello", "world", "test"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        assert tokenizer.get_vocab_size() > len(tokenizer.special_tokens)
        assert 'h' in tokenizer.vocab
        assert 'e' in tokenizer.vocab
        assert 'l' in tokenizer.vocab

    def test_encode_decode(self):
        """Test encoding and decoding."""
        texts = ["hello", "world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)

        assert decoded == "hello"

    def test_unknown_character(self):
        """Test handling of unknown characters."""
        tokenizer = CharacterTokenizer()
        tokenizer.fit(["abc"])

        # 'd' is not in vocab
        encoded = tokenizer.encode("abcd")
        assert tokenizer.special_tokens['<UNK>'] in encoded

class TestWordTokenizer:
    """Test WordTokenizer class."""

    def test_fit_and_vocab(self):
        """Test fitting and vocabulary building."""
        texts = ["hello world", "this is a test", "hello again"]
        tokenizer = WordTokenizer()
        tokenizer.fit(texts)

        assert tokenizer.get_vocab_size() > len(tokenizer.special_tokens)
        assert 'hello' in tokenizer.vocab
        assert 'world' in tokenizer.vocab

    def test_encode_decode(self):
        """Test encoding and decoding."""
        texts = ["hello world", "this is"]
        tokenizer = WordTokenizer()
        tokenizer.fit(texts)

        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)

        assert decoded == "hello world"

    def test_case_insensitive(self):
        """Test case insensitive tokenization."""
        texts = ["Hello World"]
        tokenizer = WordTokenizer(lowercase=True)
        tokenizer.fit(texts)

        encoded1 = tokenizer.encode("Hello")
        encoded2 = tokenizer.encode("hello")

        assert encoded1 == encoded2

class TestTextPreprocessor:
    """Test TextPreprocessor class."""

    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()

        text = "  Hello   World!   This is a TEST.  "
        cleaned = preprocessor.clean_text(text)

        assert cleaned == "hello world! this is a test."

    def test_pad_sequence(self):
        """Test sequence padding."""
        preprocessor = TextPreprocessor(max_length=5)

        seq = [1, 2, 3]
        padded = preprocessor.pad_sequence(seq)

        assert len(padded) == 5
        assert padded == [1, 2, 3, 0, 0]

    def test_truncate_sequence(self):
        """Test sequence truncation."""
        preprocessor = TextPreprocessor()

        seq = [1, 2, 3, 4, 5, 6]
        truncated = preprocessor.truncate_sequence(seq, 3)

        assert len(truncated) == 3
        assert truncated == [1, 2, 3]

    def test_create_sequences(self):
        """Test sequence creation for language modeling."""
        texts = ["hello world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        preprocessor = TextPreprocessor()
        X, y = preprocessor.create_sequences("hello", 3, tokenizer)

        assert X.shape[1] == 3  # sequence length
        assert y.shape[1] == 3  # target length
        assert X.shape[0] == y.shape[0]  # same number of sequences

class TestTextDataset:
    """Test TextDataset class."""

    def test_dataset_creation(self):
        """Test text dataset creation."""
        texts = ["hello world", "this is a test"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        dataset = TextDataset(texts, tokenizer, max_length=10)

        assert len(dataset) == 2
        assert len(dataset.encoded_texts) == 2
        assert len(dataset.padded_texts) == 2

    def test_getitem(self):
        """Test dataset indexing."""
        texts = ["hello"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        dataset = TextDataset(texts, tokenizer, max_length=10)

        padded, orig_len = dataset[0]
        assert len(padded) == 10
        assert orig_len == 5  # length of "hello"

    def test_create_sequences(self):
        """Test sequence creation from dataset."""
        texts = ["hello world"]
        tokenizer = CharacterTokenizer()
        tokenizer.fit(texts)

        dataset = TextDataset(texts, tokenizer)

        X, y = dataset.create_sequences(seq_length=3)

        assert X.shape[1] == 3
        assert y.shape[1] == 3

if __name__ == "__main__":
    pytest.main([__file__])