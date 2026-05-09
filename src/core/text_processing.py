"""
Text Processing - Tokenization and Text Preprocessing

Handles:
1. Text tokenization (character-level, word-level, subword)
2. Vocabulary building
3. Text encoding/decoding
4. Sequence padding and batching
5. Text cleaning and normalization
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict

class Tokenizer:
    """
    Base tokenizer class.

    Provides common tokenization functionality:
    - Text cleaning
    - Tokenization strategies
    - Vocabulary management
    - Encoding/decoding
    """

    def __init__(self, vocab_size: Optional[int] = None):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,  # Start of sequence
            '<EOS>': 3,  # End of sequence
        }

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """Convert text to token ids."""
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        """Convert token ids back to text."""
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab) + len(self.special_tokens)

class CharacterTokenizer(Tokenizer):
    """
    Character-level tokenizer.

    Tokenizes text into individual characters.
    Good for small vocabularies and morphological tasks.
    """

    def __init__(self, vocab_size: Optional[int] = None):
        super().__init__(vocab_size)
        self.chars = set()

    def fit(self, texts: List[str]):
        """Build character vocabulary."""
        for text in texts:
            self.chars.update(text)

        # Sort characters for consistent ordering
        sorted_chars = sorted(self.chars)

        # Limit vocabulary size if specified
        if self.vocab_size:
            sorted_chars = sorted_chars[:self.vocab_size - len(self.special_tokens)]

        # Build vocabularies
        self.vocab = {char: i + len(self.special_tokens)
                     for i, char in enumerate(sorted_chars)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Add special tokens to reverse vocab
        for token, idx in self.special_tokens.items():
            self.reverse_vocab[idx] = token

    def encode(self, text: str) -> List[int]:
        """Convert text to character token ids."""
        return [self.vocab.get(char, self.special_tokens['<UNK>']) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Convert character token ids back to text."""
        return ''.join([self.reverse_vocab.get(token_id, '<UNK>')
                       for token_id in token_ids])

class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer.

    Tokenizes text into words using whitespace and punctuation.
    Good for natural language processing tasks.
    """

    def __init__(self, vocab_size: Optional[int] = None, lowercase: bool = True):
        super().__init__(vocab_size)
        self.lowercase = lowercase

    def fit(self, texts: List[str]):
        """Build word vocabulary."""
        word_counts = Counter()

        for text in texts:
            if self.lowercase:
                text = text.lower()
            words = self._tokenize_words(text)
            word_counts.update(words)

        # Get most common words
        if self.vocab_size:
            max_words = self.vocab_size - len(self.special_tokens)
            most_common = word_counts.most_common(max_words)
        else:
            most_common = word_counts.most_common()

        # Build vocabularies
        self.vocab = {word: i + len(self.special_tokens)
                     for i, (word, _) in enumerate(most_common)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Add special tokens to reverse vocab
        for token, idx in self.special_tokens.items():
            self.reverse_vocab[idx] = token

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization: split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
        return words

    def encode(self, text: str) -> List[int]:
        """Convert text to word token ids."""
        if self.lowercase:
            text = text.lower()
        words = self._tokenize_words(text)
        return [self.vocab.get(word, self.special_tokens['<UNK>']) for word in words]

    def decode(self, token_ids: List[int]) -> str:
        """Convert word token ids back to text."""
        words = [self.reverse_vocab.get(token_id, '<UNK>')
                for token_id in token_ids]
        return ' '.join(words)

class TextPreprocessor:
    """
    Text preprocessing utilities.

    Handles:
    - Text cleaning
    - Normalization
    - Filtering
    - Sequence processing
    """

    def __init__(self, max_length: Optional[int] = None):
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove non-alphanumeric characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        return text.strip()

    def pad_sequence(self, sequence: List[int], max_length: Optional[int] = None,
                    pad_token: int = 0) -> List[int]:
        """Pad sequence to maximum length."""
        max_len = max_length or self.max_length
        if max_len is None:
            return sequence

        if len(sequence) > max_len:
            return sequence[:max_len]
        else:
            return sequence + [pad_token] * (max_len - len(sequence))

    def truncate_sequence(self, sequence: List[int], max_length: int) -> List[int]:
        """Truncate sequence to maximum length."""
        return sequence[:max_length]

    def create_sequences(self, text: str, seq_length: int, tokenizer: Tokenizer,
                        step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for language modeling.

        Args:
            text: Input text
            seq_length: Length of each sequence
            tokenizer: Tokenizer to use
            step: Step size for sliding window

        Returns:
            X: Input sequences (shape: n_sequences, seq_length)
            y: Target sequences (shape: n_sequences, seq_length)
        """
        token_ids = tokenizer.encode(text)

        sequences = []
        for i in range(0, len(token_ids) - seq_length, step):
            sequences.append(token_ids[i:i + seq_length + 1])

        if not sequences:
            return np.array([]), np.array([])

        sequences = np.array(sequences)
        X = sequences[:, :-1]  # Input sequences
        y = sequences[:, 1:]   # Target sequences (shifted by 1)

        return X, y

    def create_next_token_sequences(self, text: str, seq_length: int, tokenizer: Tokenizer,
                                    step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input/target sequences for next-token prediction.

        Args:
            text: Input text
            seq_length: Number of context tokens
            tokenizer: Tokenizer to use
            step: Step size for sliding window

        Returns:
            X: Input sequences (shape: n_sequences, seq_length)
            y: Target token ids (shape: n_sequences,)
        """
        token_ids = tokenizer.encode(text)
        X = []
        y = []

        for i in range(0, len(token_ids) - seq_length, step):
            X.append(token_ids[i:i + seq_length])
            y.append(token_ids[i + seq_length])

        if not X:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

class TextDataset:
    """
    Dataset class for text data.

    Handles text loading, preprocessing, and batching.
    """

    def __init__(self, texts: List[str], tokenizer: Tokenizer,
                 max_length: Optional[int] = None, preprocessor: Optional[TextPreprocessor] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or TextPreprocessor(max_length)

        # Preprocess texts
        self.processed_texts = [self.preprocessor.clean_text(text) for text in texts]

        # Encode texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.processed_texts]

        # Pad sequences
        self.padded_texts = [self.preprocessor.pad_sequence(seq, self.max_length)
                           for seq in self.encoded_texts]

    def __len__(self) -> int:
        return len(self.padded_texts)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        """Get encoded text and its original length."""
        return self.padded_texts[idx], len(self.encoded_texts[idx])

    def get_raw_text(self, idx: int) -> str:
        """Get original text."""
        return self.texts[idx]

    def get_processed_text(self, idx: int) -> str:
        """Get processed text."""
        return self.processed_texts[idx]

    def create_sequences(self, seq_length: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from all texts for language modeling.

        Returns:
            X: Input sequences
            y: Target sequences
        """
        all_X, all_y = [], []

        for text in self.processed_texts:
            X, y = self.preprocessor.create_sequences(text, seq_length, self.tokenizer, step)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            return np.array([]), np.array([])

        return np.vstack(all_X), np.vstack(all_y)

    def create_next_token_sequences(self, seq_length: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create next-token prediction sequences from all texts.

        Returns:
            X: Input sequences
            y: Next-token labels
        """
        all_X, all_y = [], []

        for text in self.processed_texts:
            X, y = self.preprocessor.create_next_token_sequences(text, seq_length, self.tokenizer, step)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            return np.array([]), np.array([])

        return np.vstack(all_X), np.hstack(all_y)

def load_text_file(filepath: str) -> List[str]:
    """
    Load text from file.

    Args:
        filepath: Path to text file

    Returns:
        List of text lines
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]

    return lines

def save_text_file(texts: List[str], filepath: str):
    """
    Save texts to file.

    Args:
        texts: List of text strings
        filepath: Output file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

# Example usage and testing
if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "Hello world!",
        "This is a test.",
        "Machine learning is fascinating.",
        "Neural networks can learn complex patterns."
    ]

    # Test character tokenizer
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.fit(sample_texts)

    print("Character Tokenizer:")
    print(f"Vocab size: {char_tokenizer.get_vocab_size()}")
    print(f"Sample encoding: {char_tokenizer.encode('hello')}")
    print(f"Sample decoding: {char_tokenizer.decode([char_tokenizer.encode('hello')[0]])}")

    # Test word tokenizer
    word_tokenizer = WordTokenizer(vocab_size=50)
    word_tokenizer.fit(sample_texts)

    print("\nWord Tokenizer:")
    print(f"Vocab size: {word_tokenizer.get_vocab_size()}")
    print(f"Sample encoding: {word_tokenizer.encode('hello world')}")
    print(f"Sample decoding: {word_tokenizer.decode(word_tokenizer.encode('hello world'))}")

    # Test text dataset
    dataset = TextDataset(sample_texts, char_tokenizer, max_length=20)
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")