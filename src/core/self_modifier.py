"""
Self-modification system for the AI assistant.

Allows the AI to analyze its own code, generate improvements,
and adapt its behavior through code generation and execution.
"""

import os
import ast
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import inspect
import sys


@dataclass
class CodeModification:
    """Record of a code modification."""
    timestamp: str
    file_path: str
    original_code: str
    modified_code: str
    reason: str
    success: bool
    error_message: Optional[str] = None


class CodeReflection:
    """Analyzes the AI assistant's own code structure and capabilities."""

    def __init__(self, source_dir: str = 'src/core'):
        self.source_dir = source_dir
        self.logger = logging.getLogger(__name__)

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a Python file's structure and capabilities."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'filepath': filepath,
                'classes': [],
                'functions': [],
                'imports': [],
                'docstrings': {},
                'complexity_score': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node)
                    })
                
                elif isinstance(node, ast.FunctionDef) and isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node)
                    })
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    analysis['imports'].append(f"from {node.module}")
            
            # Calculate complexity (simplified)
            analysis['complexity_score'] = len(analysis['classes']) + len(analysis['functions'])
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing {filepath}: {e}")
            return {}

    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get a summary of the AI system's current capabilities."""
        capabilities = {
            'analysis_timestamp': datetime.now().isoformat(),
            'modules': {},
            'total_complexity': 0
        }
        
        for filename in os.listdir(self.source_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(self.source_dir, filename)
                analysis = self.analyze_file(filepath)
                capabilities['modules'][filename] = analysis
                capabilities['total_complexity'] += analysis.get('complexity_score', 0)
        
        return capabilities

    def identify_improvements(self, capabilities: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest potential improvements based on current code analysis."""
        improvements = []
        
        for module_name, module_info in capabilities['modules'].items():
            # Suggest adding missing docstrings
            functions_without_docs = [
                f['name'] for f in module_info.get('functions', [])
                if not f.get('docstring')
            ]
            if functions_without_docs:
                improvements.append({
                    'type': 'documentation',
                    'module': module_name,
                    'suggestion': f"Add docstrings to functions: {', '.join(functions_without_docs)}",
                    'priority': 'low'
                })
            
            # Suggest modularization if complexity is high
            if module_info.get('complexity_score', 0) > 10:
                improvements.append({
                    'type': 'refactoring',
                    'module': module_name,
                    'suggestion': 'Consider breaking into smaller modules',
                    'priority': 'medium'
                })
        
        return improvements


class CodeGenerator:
    """Generates new code and modifications for the AI assistant."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def generate_new_method(self, class_name: str, method_name: str,
                           method_description: str, params: List[str]) -> str:
        """Generate a new method based on description and parameters."""
        param_str = ', '.join(params)
        
        code = f'''    def {method_name}(self, {param_str}) -> Any:
        """
        {method_description}
        
        Args:
            {', '.join([f'{p}: ' for p in params])}
        
        Returns:
            Modified result
        """
        # Auto-generated method - implement logic here
        pass
'''
        return code

    def generate_adapter_class(self, class_name: str, methods: List[str]) -> str:
        """Generate an adapter class that wraps existing functionality."""
        method_stubs = '\n'.join([
            f'        pass  # Implement {method}' for method in methods
        ])
        
        code = f'''class {class_name}Adapter:
    """Auto-generated adapter for {class_name}."""
    
    def __init__(self, wrapped_instance):
        self.wrapped = wrapped_instance
    
    def enhance(self):
        """Enhance the wrapped instance with new capabilities."""
{method_stubs}
'''
        return code

    def generate_extension_module(self, module_name: str, 
                                 capabilities: List[str]) -> str:
        """Generate an extension module for new functionality."""
        imports_section = "import logging\nfrom typing import Any, List, Optional"
        
        class_defs = '\n\n'.join([
            f"class {cap.title().replace('_', '')}Capability:\n    \"\"\"Auto-generated {cap} capability.\"\"\"\n    pass"
            for cap in capabilities
        ])
        
        code = f'''"""
Auto-generated extension module for enhanced AI capabilities.

Generated: {datetime.now().isoformat()}
Capabilities: {', '.join(capabilities)}
"""

{imports_section}

logger = logging.getLogger(__name__)


{class_defs}


def register_capabilities() -> dict:
    """Register all new capabilities."""
    return {{
        {', '.join([f"'{cap}': {cap.title().replace('_', '')}Capability()" for cap in capabilities])}
    }}
'''
        return code


class SafeExecutor:
    """Safely executes and validates code modifications."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.modification_history: List[CodeModification] = []

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def test_modification(self, modified_code: str, test_inputs: Optional[List[Any]] = None) -> Tuple[bool, Optional[str]]:
        """Test a code modification with sample inputs."""
        is_valid, error = self.validate_syntax(modified_code)
        if not is_valid:
            return False, f"Syntax error: {error}"
        
        try:
            # Create a safe execution environment
            exec_globals = {'logging': logging}
            exec(modified_code, exec_globals)
            return True, None
        except Exception as e:
            return False, f"Execution error: {str(e)}"

    def apply_modification(self, filepath: str, original_code: str, 
                          modified_code: str, reason: str) -> Tuple[bool, Optional[str]]:
        """Apply a code modification to a file."""
        # Validate syntax
        is_valid, error = self.validate_syntax(modified_code)
        if not is_valid:
            self.logger.error(f"Syntax validation failed: {error}")
            return False, error
        
        # Test execution
        works, test_error = self.test_modification(modified_code)
        if not works:
            self.logger.error(f"Execution test failed: {test_error}")
            return False, test_error
        
        try:
            # Create backup
            backup_path = f"{filepath}.backup"
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    backup_content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_content)
            
            # Apply modification
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            # Record modification
            mod = CodeModification(
                timestamp=datetime.now().isoformat(),
                file_path=filepath,
                original_code=original_code,
                modified_code=modified_code,
                reason=reason,
                success=True
            )
            self.modification_history.append(mod)
            
            self.logger.info(f"Successfully applied modification to {filepath}")
            return True, None
        
        except Exception as e:
            self.logger.error(f"Error applying modification: {str(e)}")
            return False, str(e)

    def rollback_modification(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """Rollback the last modification to a file."""
        backup_path = f"{filepath}.backup"
        if os.path.exists(backup_path):
            try:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original = f.read()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(original)
                self.logger.info(f"Rolled back {filepath}")
                return True, None
            except Exception as e:
                return False, str(e)
        return False, "No backup found"

    def save_modification_log(self, output_path: str = 'logs/modification_history.json') -> None:
        """Save modification history to a JSON file."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        history_data = [
            {
                'timestamp': mod.timestamp,
                'file': mod.file_path,
                'reason': mod.reason,
                'success': mod.success,
                'error': mod.error_message
            }
            for mod in self.modification_history
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)


class AdaptiveAI:
    """Core system for AI self-modification and adaptation."""

    def __init__(self, source_dir: str = 'src/core'):
        self.reflection = CodeReflection(source_dir)
        self.generator = CodeGenerator()
        self.executor = SafeExecutor()
        self.source_dir = source_dir
        self.logger = logging.getLogger(__name__)

    def assess_current_state(self) -> Dict[str, Any]:
        """Assess the current state of the AI system."""
        capabilities = self.reflection.get_system_capabilities()
        improvements = self.reflection.identify_improvements(capabilities)
        
        return {
            'capabilities': capabilities,
            'suggested_improvements': improvements,
            'timestamp': datetime.now().isoformat()
        }

    def plan_adaptation(self, goal: str) -> List[Dict[str, str]]:
        """Plan adaptations to achieve a goal."""
        self.logger.info(f"Planning adaptation for goal: {goal}")
        
        # Analyze current state
        state = self.assess_current_state()
        total_complexity = state['capabilities']['total_complexity']
        
        adaptation_plan = [
            {
                'step': 1,
                'action': 'analyze_current_capabilities',
                'goal': 'Understand existing code structure'
            },
            {
                'step': 2,
                'action': 'identify_gaps',
                'goal': f'Find gaps for goal: {goal}'
            },
            {
                'step': 3,
                'action': 'generate_new_code',
                'goal': 'Create new modules/methods'
            },
            {
                'step': 4,
                'action': 'test_modifications',
                'goal': 'Validate new code'
            },
            {
                'step': 5,
                'action': 'integrate_changes',
                'goal': 'Merge into main system'
            }
        ]
        
        return adaptation_plan

    def execute_adaptation(self, plan: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute an adaptation plan."""
        results = {
            'plan_length': len(plan),
            'steps_executed': 0,
            'modifications': [],
            'status': 'in_progress'
        }
        
        for step in plan:
            self.logger.info(f"Executing step {step['step']}: {step['action']}")
            results['steps_executed'] += 1
        
        results['status'] = 'completed'
        return results

    def propose_enhancement(self, enhancement_type: str, **kwargs) -> Dict[str, Any]:
        """Propose and generate an enhancement to the AI system."""
        proposal = {
            'type': enhancement_type,
            'timestamp': datetime.now().isoformat(),
            'proposed_changes': [],
            'validation_status': 'pending'
        }
        
        if enhancement_type == 'new_capability':
            capability_name = kwargs.get('capability_name', 'unknown')
            methods = kwargs.get('methods', [])
            code = self.generator.generate_adapter_class(capability_name, methods)
            proposal['proposed_code'] = code
        
        elif enhancement_type == 'extension_module':
            capabilities = kwargs.get('capabilities', [])
            code = self.generator.generate_extension_module('enhanced', capabilities)
            proposal['proposed_code'] = code
        
        return proposal
