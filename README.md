# AI Assistant from Scratch

This project builds a fully functional AI assistant completely from scratch, step-by-step, while learning AI engineering fundamentals.

## Project Goals
- Learn AI internals deeply
- Build everything manually first, then optimize
- Create a complete AI system with custom components

## Current Status
- ✅ Environment set up
- ✅ Folder structure created and scaffolded for AI/ML workflows
- ✅ Basic dependencies installed (NumPy, matplotlib, pytest)
- ✅ Manual perceptron implementation (learns AND gate)
- ✅ Multi-layer neural network (learns XOR gate)
- ✅ Backpropagation from scratch
- ✅ Binary cross-entropy loss
- ✅ Decision boundary visualization
- ✅ Activation functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
- ✅ Proper weight initialization (He/Xavier)
- ✅ Activation function comparison and analysis
- ✅ Optimization algorithms (SGD, Momentum, RMSProp, Adam)
- ✅ Optimizer comparison and performance analysis
- ✅ Production-ready training infrastructure (DataLoader, Trainer, metrics, scheduling)
- ✅ Dataset processing (synthetic data generation, preprocessing, splitting)
- ✅ Text processing (tokenization, text datasets, sequence generation)
- ✅ Embeddings and trainable text model support
- ✅ Attention architecture (scaled dot-product, multi-head self-attention, positional encoding)
- ✅ Transformer-style predictor foundation
- ✅ Transformer training integration with generic trainer
- ✅ Checkpointing for best model state
- ✅ Advanced optimizer support with learning rate scheduling and weight decay
- ✅ Sampling-based text generation
- ✅ Mini language model wrapper and training interface
- ✅ Assistant system with conversation memory and context-aware prompts
- ✅ Production assistant wrapper with configurable prompts, checkpointing, and knowledge retrieval
- ✅ Voice interface fallback with optional pyttsx3/espeak support

## Recent Achievements
- **Training Infrastructure**: Complete training pipeline with batching, validation, early stopping, and learning rate scheduling
- **Dataset Processing**: Synthetic dataset generation (XOR, circles, moons, blobs), data preprocessing, and cross-validation
- **Text Processing**: Character and word-level tokenization, text cleaning, sequence generation for language modeling
- **Attention/Transformers**: Added core attention blocks, positional encoding, transformer encoder layer, and a minimal transformer predictor

## Roadmap
1. ✅ Python mastery
2. ✅ NumPy and matrix operations
3. ✅ Build a perceptron manually
4. ✅ Build multi-layer neural networks manually
5. ✅ Activation functions
6. ✅ Gradient descent
7. ✅ Backpropagation
8. ✅ Loss functions
9. ✅ Training loops
10. ✅ Dataset processing
11. ✅ Build a tiny text predictor
12. ✅ Tokenization
13. ✅ Embeddings
14. ✅ Attention
15. ✅ Transformers
16. ✅ Mini language model
17. ✅ Assistant systems
18. ✅ Voice systems
19. ✅ Memory systems
20. ✅ Optimization

## Folder Structure
```
AI/
├── .git/
├── .gitignore
├── .pytest_cache/
├── README.md
├── checkpoints/
├── configs/
├── data_memory/
├── datasets/
├── docs/
├── experiments/
│   ├── demo_assistant_cli.py
│   ├── demo_assistant_production.py
│   ├── demo_datasets_and_text.py
│   ├── demo_text_training.py
│   ├── demo_transformer_training.py
│   └── evaluate_assistant.py
├── logs/
├── models/
├── notebooks/
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── neural_networks/
│   │   │   ├── perceptron.py      # Single neuron classifier
│   │   │   ├── mlp.py             # Multi-layer perceptron
│   │   │   ├── activations.py     # Activation functions
│   │   │   └── optimizers.py      # Optimization algorithms
│   │   ├── training.py            # Training infrastructure
│   │   ├── datasets.py            # Dataset processing
│   │   ├── text_processing.py     # Text tokenization & processing
│   │   ├── text_model.py          # Tiny text predictor model
│   │   ├── attention/
│   │   │   └── attention.py       # Attention building blocks
│   │   └── transformers/
│   │       └── transformer.py     # Minimal transformer predictor
│   ├── utils/
│   └── models/
├── tests/
│   ├── test_perceptron.py
│   ├── test_mlp.py
│   ├── test_activations.py
│   ├── test_optimizers.py
│   ├── test_datasets_and_text.py
│   ├── test_text_model.py
│   ├── test_attention_transformer.py
│   └── test_transformer_training.py
└── venv/
```

## Getting Started
1. Ensure Python 3.12+ is installed
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) For voice output, install `pyttsx3`
4. Follow the learning roadmap step-by-step

## Evaluation
Run the evaluation script to test the assistant's performance:
```bash
python experiments/evaluate_assistant.py
```
This will evaluate the assistant on a set of test prompts and generate statistics about response quality.