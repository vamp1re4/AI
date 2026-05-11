"""
Extended Training Script for Self-Modifying AI Assistant

This script implements a comprehensive training regimen with:
- Large dataset generation
- Progressive curriculum learning
- Extended training epochs
- Performance monitoring
- Checkpointing and recovery
- Self-improvement integration
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.adaptive_assistant import AdaptiveAssistant
from src.core.text_processing import CharacterTokenizer
from src.core.language_model import MiniLanguageModel
from src.core.production import ProductionAssistantConfig


class ExtendedTrainer:
    """Extended training system for the AI assistant."""

    def __init__(self, max_iterations: int = 100000):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.start_time = None
        self.training_log = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 10

        # Training curriculum
        self.curriculum_stages = [
            {'name': 'basic_patterns', 'epochs': 50, 'complexity': 0.1},
            {'name': 'conversational', 'epochs': 100, 'complexity': 0.3},
            {'name': 'technical', 'epochs': 150, 'complexity': 0.5},
            {'name': 'advanced', 'epochs': 200, 'complexity': 0.8},
            {'name': 'expert', 'epochs': 500, 'complexity': 1.0}
        ]

        self.current_stage = 0

    def generate_training_data(self, stage: int, size: int = 10000) -> List[str]:
        """Generate training data for different curriculum stages."""
        stage_info = self.curriculum_stages[min(stage, len(self.curriculum_stages) - 1)]
        complexity = stage_info['complexity']

        base_patterns = [
            "Hello, how are you?",
            "I am doing well, thank you.",
            "What can you help me with?",
            "I can assist with various tasks.",
            "Tell me more about that.",
            "That's interesting to know.",
            "How does this work?",
            "Let me explain it to you.",
            "What are your capabilities?",
            "I can learn and adapt over time."
        ]

        if complexity > 0.3:
            base_patterns.extend([
                "The AI system uses neural networks.",
                "Machine learning involves training models.",
                "Transformers are powerful for language tasks.",
                "Self-modification allows continuous improvement.",
                "Code analysis helps identify optimization opportunities.",
                "Feedback drives adaptation and learning.",
                "The system can generate new capabilities.",
                "Continuous learning is key to intelligence.",
                "Performance monitoring tracks improvement.",
                "The AI can analyze its own code structure."
            ])

        if complexity > 0.5:
            base_patterns.extend([
                "Gradient descent optimizes neural network parameters.",
                "Backpropagation computes gradients through the network.",
                "Attention mechanisms focus on relevant information.",
                "Positional encoding provides sequence position information.",
                "Layer normalization stabilizes training dynamics.",
                "Dropout prevents overfitting during training.",
                "Batch normalization accelerates convergence.",
                "Regularization techniques improve generalization.",
                "Hyperparameter tuning optimizes model performance.",
                "Cross-validation evaluates model robustness."
            ])

        if complexity > 0.8:
            base_patterns.extend([
                "The transformer architecture revolutionized natural language processing.",
                "Self-supervised learning leverages unlabeled data effectively.",
                "Meta-learning enables rapid adaptation to new tasks.",
                "Neural architecture search automates model design.",
                "Federated learning preserves privacy in distributed settings.",
                "Adversarial training improves model robustness.",
                "Knowledge distillation compresses large models efficiently.",
                "Multi-task learning shares knowledge across domains.",
                "Continual learning prevents catastrophic forgetting.",
                "Explainable AI provides insights into model decisions."
            ])

        # Generate variations and combinations
        training_data = base_patterns.copy()

        # Add variations with different phrasing
        for pattern in base_patterns[:50]:  # Limit to avoid explosion
            words = pattern.split()
            if len(words) > 3:
                # Create variations
                variations = [
                    f"I think {pattern.lower()}",
                    f"Do you know {pattern.lower()}",
                    f"That's because {pattern.lower()}",
                    f"Well, {pattern.lower()}",
                    f"Actually, {pattern.lower()}"
                ]
                training_data.extend(variations)

        # Add question-answer pairs
        qa_pairs = [
            ("What is AI?", "Artificial Intelligence is the simulation of human intelligence in machines."),
            ("How do neural networks work?", "Neural networks process information through interconnected nodes called neurons."),
            ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."),
            ("How does backpropagation work?", "Backpropagation calculates gradients to update neural network weights."),
            ("What are transformers?", "Transformers are neural network architectures that excel at sequence processing."),
        ]

        for question, answer in qa_pairs:
            training_data.extend([question, answer])

        # Ensure we have enough data
        while len(training_data) < size:
            # Generate synthetic combinations
            idx1, idx2 = np.random.randint(0, len(base_patterns), 2)
            combined = f"{base_patterns[idx1]} {base_patterns[idx2]}"
            training_data.append(combined)

        # Shuffle and return
        np.random.shuffle(training_data)
        return training_data[:size]

    def setup_training_environment(self):
        """Set up the training environment."""
        print("🚀 Setting up Extended AI Training Environment...")

        # Create directories
        os.makedirs('checkpoints/extended_training', exist_ok=True)
        os.makedirs('logs/extended_training', exist_ok=True)
        os.makedirs('models/extended_training', exist_ok=True)

        # Initialize tokenizer with comprehensive vocabulary
        print("📚 Building comprehensive tokenizer...")
        initial_texts = self.generate_training_data(0, 1000)
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.fit(initial_texts)

        print(f"🔤 Vocabulary size: {self.tokenizer.get_vocab_size()}")

        # Initialize model with optimized parameters
        self.model = MiniLanguageModel(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=128,  # Larger model
            num_heads=8,
            d_ff=256,
            num_layers=4,  # Deeper model
            max_seq_len=64,  # Longer sequences
            optimizer='adam',
            optimizer_params={'weight_decay': 1e-4}
        )

        # Configure training
        self.config = ProductionAssistantConfig(
            enable_self_modification=True,
            training_epochs=100,  # Will be overridden per stage
            training_batch_size=16,
            training_val_split=0.15,
            training_seq_length=32,
            training_patience=15,
            max_seq_len=64
        )

        # Initialize assistant
        self.assistant = AdaptiveAssistant(
            tokenizer=self.tokenizer,
            language_model=self.model,
            config=self.config,
            learning_enabled=True
        )

        print("✅ Training environment ready!")

    def train_curriculum_stage(self, stage_idx: int) -> Dict[str, Any]:
        """Train a specific curriculum stage."""
        stage = self.curriculum_stages[stage_idx]
        print(f"\n🎯 Training Stage {stage_idx + 1}: {stage['name']} (Complexity: {stage['complexity']})")

        # Generate training data for this stage
        print("📝 Generating training data...")
        training_texts = self.generate_training_data(stage_idx, 5000)
        print(f"📊 Generated {len(training_texts)} training samples")

        # Update config for this stage
        stage_config = ProductionAssistantConfig(
            enable_self_modification=self.config.enable_self_modification,
            training_epochs=stage['epochs'],
            training_batch_size=self.config.training_batch_size,
            training_val_split=self.config.training_val_split,
            training_seq_length=self.config.training_seq_length,
            training_patience=self.config.training_patience,
            max_seq_len=self.config.max_seq_len
        )
        stage_config.training_epochs = stage['epochs']

        # Train the model
        print(f"🏃 Training for {stage['epochs']} epochs...")
        start_time = time.time()

        train_result = self.assistant.train(training_texts, verbose=False)

        training_time = time.time() - start_time
        final_loss = train_result.train_losses[-1]
        best_val_loss = min(train_result.val_losses)

        # Log results
        stage_result = {
            'stage': stage_idx,
            'stage_name': stage['name'],
            'complexity': stage['complexity'],
            'epochs_completed': len(train_result.train_losses),
            'final_train_loss': final_loss,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'samples_used': len(training_texts),
            'timestamp': datetime.now().isoformat()
        }

        self.training_log.append(stage_result)

        print(".4f")
        print(".4f")
        print(".2f")
        # Save checkpoint
        checkpoint_path = f'models/extended_training/stage_{stage_idx}_{stage["name"]}.npz'
        self.model.save(checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Test the model
        self.test_model_performance(stage_idx)

        return stage_result

    def test_model_performance(self, stage_idx: int):
        """Test model performance at current stage."""
        print("🧪 Testing model performance...")

        test_prompts = [
            "Hello",
            "What is AI",
            "How do you learn",
            "Tell me about neural networks",
            "Explain machine learning"
        ]

        print("Sample generations:")
        for prompt in test_prompts[:3]:  # Test first 3
            try:
                tokens = self.tokenizer.encode(prompt)
                generated = self.model.generate(
                    initial_tokens=np.array(tokens),
                    max_new_tokens=20,
                    temperature=0.8,
                    sample=True
                )
                response = self.tokenizer.decode(generated[len(tokens):].tolist())
                clean_response = response.replace('<PAD>', '').replace('<UNK>', '').strip()
                print(f"  '{prompt}' → '{clean_response[:40]}...'")
            except Exception as e:
                print(f"  '{prompt}' → Error: {e}")

    def run_extended_training(self):
        """Run the complete extended training curriculum."""
        print("="*80)
        print("🤖 EXTENDED AI TRAINING REGIMEN")
        print("="*80)
        print(f"Target: {self.max_iterations} iterations across curriculum stages")
        print(f"Stages: {len(self.curriculum_stages)}")
        print(f"Start Time: {datetime.now()}")
        print("="*80)

        self.start_time = time.time()

        try:
            for stage_idx in range(len(self.curriculum_stages)):
                # Train this stage
                stage_result = self.train_curriculum_stage(stage_idx)

                # Check if we should continue
                if self.current_iteration >= self.max_iterations:
                    print(f"\n🎯 Reached target iterations: {self.max_iterations}")
                    break

                # Self-improvement phase
                print("\n🚀 Self-Improvement Phase...")
                improvement = self.assistant.auto_improve(f'stage_{stage_idx}_optimization')
                print(f"   Status: {improvement['status']}")

                # Save progress
                self.save_training_progress()

                print(f"\n✅ Stage {stage_idx + 1} completed!")

            print("\n🎉 Extended training completed!")
            self.final_evaluation()

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            self.save_training_progress()
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            self.save_training_progress()

    def final_evaluation(self):
        """Final evaluation of the trained model."""
        print("\n" + "="*60)
        print("🎯 FINAL EVALUATION")
        print("="*60)

        # Load best model
        best_checkpoint = self.find_best_checkpoint()
        if best_checkpoint:
            print(f"📁 Loading best model: {best_checkpoint}")
            self.model.load(best_checkpoint)

        # Comprehensive testing
        test_results = self.comprehensive_test()

        # Self-assessment
        assessment = self.assistant.get_self_assessment()
        health = self.assistant.get_system_health()

        # Generate final report
        final_report = {
            'training_summary': {
                'total_stages': len(self.training_log),
                'total_time': time.time() - self.start_time,
                'best_val_loss': min([s['best_val_loss'] for s in self.training_log]),
                'final_assessment': assessment,
                'system_health': health
            },
            'stage_results': self.training_log,
            'test_results': test_results,
            'completion_timestamp': datetime.now().isoformat()
        }

        # Save final report
        with open('logs/extended_training/final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        print("📄 Final report saved to logs/extended_training/final_report.json")

        # Display summary
        print("\n🏆 TRAINING SUMMARY:")
        print(f"   Stages Completed: {len(self.training_log)}")
        print(".2f")
        print(".4f")
        print(f"   System Health: {health['status']}")
        print(f"   Self-Modifications: {health['modifications_applied']}")

    def comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive tests on the trained model."""
        print("🧪 Running comprehensive evaluation...")

        test_categories = {
            'basic_conversation': [
                "Hello", "How are you?", "What is your name?",
                "Thank you", "Goodbye", "Nice to meet you"
            ],
            'technical_questions': [
                "What is AI?", "How do neural networks work?",
                "Explain machine learning", "What are transformers?",
                "How does backpropagation work?"
            ],
            'learning_questions': [
                "How do you learn?", "Can you improve yourself?",
                "What can you do?", "Tell me about your capabilities",
                "How do you adapt?"
            ],
            'complex_topics': [
                "Explain gradient descent", "What is self-modification?",
                "How does attention work?", "Describe federated learning",
                "What is meta-learning?"
            ]
        }

        results = {}

        for category, prompts in test_categories.items():
            print(f"  Testing {category}...")
            category_results = []

            for prompt in prompts:
                try:
                    tokens = self.tokenizer.encode(prompt)
                    generated = self.model.generate(
                        initial_tokens=np.array(tokens),
                        max_new_tokens=30,
                        temperature=0.7,
                        sample=True
                    )
                    response = self.tokenizer.decode(generated[len(tokens):].tolist())
                    clean_response = response.replace('<PAD>', '').replace('<UNK>', '').strip()

                    # Basic quality metrics
                    response_length = len(clean_response.split())
                    has_question_mark = '?' in clean_response
                    has_period = '.' in clean_response

                    category_results.append({
                        'prompt': prompt,
                        'response': clean_response[:100],
                        'length': response_length,
                        'has_punctuation': has_question_mark or has_period
                    })

                except Exception as e:
                    category_results.append({
                        'prompt': prompt,
                        'error': str(e)
                    })

            results[category] = category_results

        return results

    def find_best_checkpoint(self) -> str:
        """Find the best checkpoint based on validation loss."""
        model_dir = 'models/extended_training'
        if not os.path.exists(model_dir):
            return None

        checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.npz')]
        if not checkpoints:
            return None

        # For now, return the latest checkpoint
        # In a real implementation, you'd track validation losses
        return os.path.join(model_dir, sorted(checkpoints)[-1])

    def save_training_progress(self):
        """Save current training progress."""
        progress = {
            'current_stage': self.current_stage,
            'current_iteration': self.current_iteration,
            'training_log': self.training_log,
            'timestamp': datetime.now().isoformat(),
            'total_time': time.time() - self.start_time if self.start_time else 0
        }

        with open('logs/extended_training/progress.json', 'w') as f:
            json.dump(progress, f, indent=2, default=str)


def main():
    """Main training function."""
    print("🚀 Starting Extended AI Training Program")
    print("This will train the AI through multiple curriculum stages")
    print("Estimated time: Several hours to days depending on hardware")
    print()

    # Create trainer
    trainer = ExtendedTrainer(max_iterations=100000)  # Reasonable limit

    # Setup environment
    trainer.setup_training_environment()

    # Run extended training
    trainer.run_extended_training()

    print("\n🎉 Extended training program completed!")
    print("Check logs/extended_training/ for detailed results")


if __name__ == '__main__':
    main()
