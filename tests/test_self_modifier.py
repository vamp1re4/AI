"""
Tests for the self-modification and adaptive learning system.

Tests cover:
- Code reflection and analysis
- Code generation
- Safe execution and validation
- Adaptive AI planning
- Integration with assistant systems
"""

import pytest
import tempfile
import os
import json
from pathlib import Path

from src.core.self_modifier import (
    CodeReflection,
    CodeGenerator,
    SafeExecutor,
    AdaptiveAI,
    CodeModification
)
from src.core.adaptive_assistant import AdaptiveAssistant
from src.core.text_processing import Tokenizer
from src.core.language_model import MiniLanguageModel


class TestCodeReflection:
    """Test code analysis and reflection capabilities."""

    def test_analyze_file(self):
        """Test analyzing a Python file."""
        reflection = CodeReflection()
        
        # Use a real file in the project
        test_file = 'src/core/text_processing.py'
        if os.path.exists(test_file):
            analysis = reflection.analyze_file(test_file)
            
            assert 'filepath' in analysis
            assert 'classes' in analysis
            assert 'functions' in analysis
            assert 'imports' in analysis
            assert 'complexity_score' in analysis

    def test_identify_improvements(self):
        """Test improvement identification."""
        reflection = CodeReflection()
        
        # Create mock capabilities
        capabilities = {
            'modules': {
                'test_module.py': {
                    'classes': [],
                    'functions': [
                        {'name': 'func1', 'docstring': None},
                        {'name': 'func2', 'docstring': 'Documented'},
                    ],
                    'complexity_score': 15
                }
            }
        }
        
        improvements = reflection.identify_improvements(capabilities)
        
        assert len(improvements) > 0
        assert any(imp['type'] == 'documentation' for imp in improvements)


class TestCodeGenerator:
    """Test code generation capabilities."""

    def test_generate_method(self):
        """Test generating a new method."""
        generator = CodeGenerator()
        
        code = generator.generate_new_method(
            'TestClass',
            'test_method',
            'A test method',
            ['param1', 'param2']
        )
        
        assert 'def test_method' in code
        assert 'param1' in code
        assert 'param2' in code
        assert 'docstring' not in code or '"""' in code

    def test_generate_adapter_class(self):
        """Test generating adapter class."""
        generator = CodeGenerator()
        
        code = generator.generate_adapter_class(
            'TestClass',
            ['method1', 'method2']
        )
        
        assert 'class TestClassAdapter' in code
        assert 'method1' in code
        assert 'method2' in code

    def test_generate_extension_module(self):
        """Test generating extension module."""
        generator = CodeGenerator()
        
        code = generator.generate_extension_module(
            'enhanced_core',
            ['capability1', 'capability2', 'capability3']
        )
        
        assert 'import logging' in code
        assert 'from typing import' in code
        assert 'capability1' in code
        assert 'def register_capabilities' in code


class TestSafeExecutor:
    """Test safe code execution and validation."""

    def test_validate_syntax_valid(self):
        """Test syntax validation with valid code."""
        executor = SafeExecutor()
        
        valid_code = """
def hello():
    return "world"
"""
        is_valid, error = executor.validate_syntax(valid_code)
        assert is_valid is True
        assert error is None

    def test_validate_syntax_invalid(self):
        """Test syntax validation with invalid code."""
        executor = SafeExecutor()
        
        invalid_code = """
def hello(
    return "world"
"""
        is_valid, error = executor.validate_syntax(invalid_code)
        assert is_valid is False
        assert error is not None

    def test_test_modification(self):
        """Test modification testing."""
        executor = SafeExecutor()
        
        valid_code = """
class TestClass:
    def __init__(self):
        self.value = 42
"""
        works, error = executor.test_modification(valid_code)
        assert works is True
        assert error is None

    def test_apply_modification(self):
        """Test applying a modification."""
        executor = SafeExecutor()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            original = "# Original code\nprint('hello')"
            f.write(original)
            f.flush()
            
            filepath = f.name
        
        try:
            modified = "# Modified code\nprint('world')"
            success, error = executor.apply_modification(
                filepath,
                original,
                modified,
                'Test modification'
            )
            
            assert success is True
            assert error is None
            
            # Verify file was modified
            with open(filepath, 'r') as f:
                content = f.read()
            assert content == modified
            
            # Verify backup exists
            assert os.path.exists(f"{filepath}.backup")
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(f"{filepath}.backup"):
                os.remove(f"{filepath}.backup")

    def test_rollback_modification(self):
        """Test rolling back a modification."""
        executor = SafeExecutor()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            original = "# Original"
            f.write(original)
            f.flush()
            filepath = f.name
        
        try:
            # Create backup
            backup_path = f"{filepath}.backup"
            with open(backup_path, 'w') as f:
                f.write(original)
            
            # Modify file
            with open(filepath, 'w') as f:
                f.write("# Modified")
            
            # Rollback
            success, error = executor.rollback_modification(filepath)
            assert success is True
            
            # Verify content is restored
            with open(filepath, 'r') as f:
                content = f.read()
            assert content == original
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(f"{filepath}.backup"):
                os.remove(f"{filepath}.backup")


class TestAdaptiveAI:
    """Test the main AdaptiveAI system."""

    def test_assess_current_state(self):
        """Test current state assessment."""
        ai = AdaptiveAI()
        state = ai.assess_current_state()
        
        assert 'capabilities' in state
        assert 'suggested_improvements' in state
        assert 'timestamp' in state

    def test_plan_adaptation(self):
        """Test adaptation planning."""
        ai = AdaptiveAI()
        plan = ai.plan_adaptation("test_goal")
        
        assert len(plan) > 0
        assert all('step' in step and 'action' in step for step in plan)

    def test_execute_adaptation(self):
        """Test executing adaptation plan."""
        ai = AdaptiveAI()
        plan = ai.plan_adaptation("test_goal")
        results = ai.execute_adaptation(plan)
        
        assert 'status' in results
        assert results['status'] == 'completed'
        assert 'steps_executed' in results

    def test_propose_enhancement(self):
        """Test enhancement proposal."""
        ai = AdaptiveAI()
        
        # Test new capability
        proposal = ai.propose_enhancement(
            'new_capability',
            capability_name='TestCapability',
            methods=['method1', 'method2']
        )
        
        assert proposal['type'] == 'new_capability'
        assert 'proposed_code' in proposal
        assert 'TestCapability' in proposal['proposed_code']


class TestAdaptiveAssistant:
    """Test the AdaptiveAssistant system."""

    @pytest.fixture
    def assistant(self):
        """Create a test assistant instance."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(['hello', 'world', 'test', 'improve', 'code'])
        
        model = MiniLanguageModel(
            vocab_size=tokenizer.get_vocab_size(),
            max_seq_len=32
        )
        
        return AdaptiveAssistant(
            tokenizer=tokenizer,
            language_model=model,
            learning_enabled=True
        )

    def test_assistant_initialization(self, assistant):
        """Test assistant initialization."""
        assert assistant is not None
        assert assistant.learning_enabled is True
        assert assistant.adaptive_ai is not None

    def test_get_self_assessment(self, assistant):
        """Test self-assessment."""
        assessment = assistant.get_self_assessment()
        
        assert 'capabilities' in assessment
        assert 'interaction_metrics' in assessment
        assert 'recent_insights' in assessment

    def test_learn_from_feedback(self, assistant):
        """Test learning from feedback."""
        result = assistant.learn_from_feedback("too formal")
        
        assert 'status' in result
        assert 'actions_taken' in result
        assert result['status'] == 'completed'

    def test_get_system_health(self, assistant):
        """Test system health check."""
        health = assistant.get_system_health()
        
        assert 'status' in health
        assert 'uptime_interactions' in health
        assert 'learning_enabled' in health
        assert 'adaptive_ai_active' in health

    def test_auto_improve(self, assistant):
        """Test auto-improvement."""
        result = assistant.auto_improve(target_area='test')
        
        assert 'status' in result
        assert 'goal' in result


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_learning_cycle(self):
        """Test complete learning and improvement cycle."""
        # Initialize
        tokenizer = Tokenizer()
        tokenizer.build_vocab(['test', 'code', 'learn', 'improve'])
        model = MiniLanguageModel(vocab_size=tokenizer.get_vocab_size())
        assistant = AdaptiveAssistant(tokenizer, model, learning_enabled=True)
        
        # Check initial state
        health1 = assistant.get_system_health()
        assert health1['status'] == 'initializing'
        
        # Interact
        assessment = assistant.get_self_assessment()
        assert 'capabilities' in assessment
        
        # Learn and improve
        improvement = assistant.auto_improve('test_goal')
        assert improvement['status'] == 'completed'

    def test_capability_inspection_flow(self):
        """Test inspecting capabilities."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(['test'])
        model = MiniLanguageModel(vocab_size=tokenizer.get_vocab_size())
        assistant = AdaptiveAssistant(tokenizer, model)
        
        # Get capabilities
        capabilities = assistant.assess_capabilities()
        assert 'modules' in capabilities
        
        # Get info
        info = assistant.get_capabilities_info()
        assert 'ADAPTIVE ASSISTANT' in info
        assert 'Status' in info


# Parametrized tests for multiple scenarios
@pytest.mark.parametrize("feedback,expected_action", [
    ("too formal", "Adjust formality level"),
    ("too technical", "Simplify explanations"),
    ("more detail", "Expand information"),
    ("less detail", "Reduce verbosity"),
])
def test_feedback_responses(feedback, expected_action):
    """Test various feedback scenarios."""
    tokenizer = Tokenizer()
    tokenizer.build_vocab(['test'])
    model = MiniLanguageModel(vocab_size=tokenizer.get_vocab_size())
    assistant = AdaptiveAssistant(tokenizer, model)
    
    result = assistant.learn_from_feedback(feedback)
    assert expected_action in result['actions_taken']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
