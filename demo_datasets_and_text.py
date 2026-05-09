"""
Demonstration of Dataset Processing and Text Processing capabilities.

This script shows how to:
1. Generate synthetic datasets
2. Preprocess data
3. Split data for training/validation/testing
4. Tokenize and process text
5. Create text datasets for language modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core.datasets import DatasetGenerator, Preprocessor, DataSplitter, load_dataset
from src.core.text_processing import CharacterTokenizer, WordTokenizer, TextDataset, TextPreprocessor

def demonstrate_datasets():
    """Demonstrate dataset generation and processing."""
    print("=== Dataset Processing Demo ===\n")

    # Generate XOR dataset
    print("1. Generating XOR dataset...")
    xor_dataset = DatasetGenerator.make_xor(n_samples=200, noise=0.1)
    print(f"   Dataset: {xor_dataset.name}")
    print(f"   Samples: {len(xor_dataset)}")
    print(f"   Features: {xor_dataset.n_features}")

    # Show dataset statistics
    stats = xor_dataset.get_stats()
    print(f"   X range: [{stats['X_min']:.2f}, {stats['X_max']:.2f}]")
    print(f"   Classes: {stats['y_unique']}")

    # Preprocess the data
    print("\n2. Preprocessing data...")
    preprocessor = Preprocessor()
    X_processed, y_processed = preprocessor.fit_transform(xor_dataset.X, xor_dataset.y)

    print("   Original X mean/std: {:.3f}/{:.3f}".format(
        np.mean(xor_dataset.X), np.std(xor_dataset.X)))
    print("   Processed X mean/std: {:.3f}/{:.3f}".format(
        np.mean(X_processed), np.std(X_processed)))

    # Split the data
    print("\n3. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.train_val_test_split(
        X_processed, y_processed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Generate other datasets
    print("\n4. Generating other datasets...")
    datasets = {
        'circles': DatasetGenerator.make_circles(n_samples=150),
        'moons': DatasetGenerator.make_moons(n_samples=150),
        'blobs': DatasetGenerator.make_blobs(n_samples=150, centers=3)
    }

    for name, dataset in datasets.items():
        print(f"   {name.capitalize()}: {len(dataset)} samples, {dataset.n_features} features")

    return X_train, y_train, X_val, y_val, X_test, y_test

def demonstrate_text_processing():
    """Demonstrate text processing and tokenization."""
    print("\n=== Text Processing Demo ===\n")

    # Sample text data
    sample_texts = [
        "Hello world! This is a test.",
        "Machine learning is fascinating.",
        "Neural networks can learn complex patterns.",
        "Deep learning uses multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to see and interpret images.",
        "Reinforcement learning teaches agents through trial and error.",
        "Transfer learning applies knowledge from one task to another."
    ]

    print("1. Sample texts:")
    for i, text in enumerate(sample_texts[:3], 1):
        print(f"   {i}. {text}")
    print("   ...")

    # Character-level tokenization
    print("\n2. Character-level tokenization...")
    char_tokenizer = CharacterTokenizer()
    char_tokenizer.fit(sample_texts)

    print(f"   Vocabulary size: {char_tokenizer.get_vocab_size()}")
    print(f"   Special tokens: {list(char_tokenizer.special_tokens.keys())}")

    test_text = "hello"
    encoded = char_tokenizer.encode(test_text)
    decoded = char_tokenizer.decode(encoded)
    print(f"   Original: '{test_text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")

    # Word-level tokenization
    print("\n3. Word-level tokenization...")
    word_tokenizer = WordTokenizer(vocab_size=50)
    word_tokenizer.fit(sample_texts)

    print(f"   Vocabulary size: {word_tokenizer.get_vocab_size()}")
    print(f"   Top words: {list(word_tokenizer.vocab.keys())[:10]}")

    test_sentence = "machine learning"
    encoded = word_tokenizer.encode(test_sentence)
    decoded = word_tokenizer.decode(encoded)
    print(f"   Original: '{test_sentence}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")

    # Text preprocessing
    print("\n4. Text preprocessing...")
    preprocessor = TextPreprocessor(max_length=20)

    messy_text = "  HELLO   world!!!   This is a TEST.  "
    cleaned = preprocessor.clean_text(messy_text)
    print(f"   Original: '{messy_text}'")
    print(f"   Cleaned: '{cleaned}'")

    # Create text dataset
    print("\n5. Creating text dataset...")
    text_dataset = TextDataset(sample_texts, char_tokenizer, max_length=50)

    print(f"   Dataset size: {len(text_dataset)}")
    print(f"   Sample item: encoded length = {len(text_dataset.encoded_texts[0])}, padded length = {len(text_dataset.padded_texts[0])}")

    # Create sequences for language modeling
    print("\n6. Creating sequences for language modeling...")
    X_seq, y_seq = text_dataset.create_sequences(seq_length=10, step=3)

    print(f"   Total sequences: {len(X_seq)}")
    print(f"   Sequence shape: {X_seq.shape}")
    print(f"   Sample input sequence: {X_seq[0]}")
    print(f"   Sample target sequence: {y_seq[0]}")

    # Decode a sequence
    decoded_input = char_tokenizer.decode(X_seq[0])
    decoded_target = char_tokenizer.decode(y_seq[0])
    print(f"   Decoded input: '{decoded_input}'")
    print(f"   Decoded target: '{decoded_target}'")

    return text_dataset, X_seq, y_seq

def demonstrate_cross_validation():
    """Demonstrate k-fold cross-validation."""
    print("\n=== Cross-Validation Demo ===\n")

    # Generate a dataset
    dataset = DatasetGenerator.make_blobs(n_samples=100, centers=3, random_state=42)
    X, y = dataset.X, dataset.y.flatten()

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")

    # Perform 5-fold cross-validation
    folds = DataSplitter.k_fold_split(X, y, k=5, random_state=42)

    print(f"\n5-fold cross-validation:")
    for i, (X_train, X_val, y_train, y_val) in enumerate(folds, 1):
        print(f"   Fold {i}: Train={len(X_train)}, Val={len(X_val)}")
        print(f"       Train classes: {np.unique(y_train)}")
        print(f"       Val classes: {np.unique(y_val)}")

def main():
    """Main demonstration function."""
    print("AI Assistant - Dataset and Text Processing Demo")
    print("=" * 50)

    # Demonstrate dataset processing
    X_train, y_train, X_val, y_val, X_test, y_test = demonstrate_datasets()

    # Demonstrate text processing
    text_dataset, X_seq, y_seq = demonstrate_text_processing()

    # Demonstrate cross-validation
    demonstrate_cross_validation()

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nKey takeaways:")
    print("• Generated synthetic datasets (XOR, circles, moons, blobs)")
    print("• Implemented data preprocessing (standardization)")
    print("• Created train/val/test splits and cross-validation folds")
    print("• Built character-level and word-level tokenizers")
    print("• Created text datasets with sequence generation for language modeling")
    print("• Demonstrated text cleaning and preprocessing")

    # Show data shapes for neural network training
    print("\nData shapes ready for neural network training:")
    print(f"• XOR dataset: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"• Text sequences: X_seq={X_seq.shape}, y_seq={y_seq.shape}")

if __name__ == "__main__":
    main()