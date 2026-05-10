"""
Advanced adaptive assistant with self-modification and learning capabilities.

This assistant can analyze its own code, propose improvements, and adapt
its behavior based on interactions and feedback.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .production import ProductionAssistant, ProductionAssistantConfig
from .self_modifier import AdaptiveAI, CodeReflection, CodeGenerator, SafeExecutor
from .text_processing import Tokenizer
from .language_model import MiniLanguageModel
from .memory import ConversationMemory, KnowledgeMemory


class AdaptiveAssistant(ProductionAssistant):
    """
    Advanced assistant that can self-modify and learn from interactions.
    
    Capabilities:
    - Analyze its own code structure
    - Propose enhancements and new features
    - Track conversation patterns and adapt responses
    - Generate new methods and classes
    - Maintain a history of improvements
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 language_model: MiniLanguageModel,
                 config: Optional[ProductionAssistantConfig] = None,
                 memory: Optional[ConversationMemory] = None,
                 knowledge: Optional[KnowledgeMemory] = None,
                 learning_enabled: bool = True):
        """Initialize adaptive assistant with learning capabilities."""
        config = config or ProductionAssistantConfig(enable_self_modification=True)
        config.enable_self_modification = True
        
        super().__init__(
            tokenizer=tokenizer,
            language_model=language_model,
            config=config,
            memory=memory,
            knowledge=knowledge
        )
        
        self.learning_enabled = learning_enabled
        self.logger = logging.getLogger(__name__)
        
        # Interaction analytics
        self.interaction_count = 0
        self.adaptation_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_responses': 0,
            'total_improvements': 0,
            'avg_response_quality': 0.0
        }

    def respond(self, user_text: str) -> str:
        """Generate response and analyze for learning opportunities."""
        response = super().respond(user_text)
        
        self.interaction_count += 1
        self.performance_metrics['total_responses'] += 1
        
        # Analyze interaction for learning
        if self.learning_enabled:
            self._analyze_interaction(user_text, response)
        
        return response

    def _analyze_interaction(self, user_text: str, response: str) -> None:
        """Analyze interaction to identify improvement opportunities."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'user_text': user_text,
            'response': response,
            'insights': []
        }
        
        # Check response quality
        response_length = len(response.split())
        if response_length < 5:
            analysis['insights'].append('Response too short - consider elaborating')
        elif response_length > 100:
            analysis['insights'].append('Response too long - consider being more concise')
        
        # Check for common patterns
        if any(keyword in user_text.lower() for keyword in ['improve', 'enhance', 'better']):
            analysis['insights'].append('User is asking for improvements - analyze capabilities')
        
        self.adaptation_history.append(analysis)

    def get_self_assessment(self) -> Dict[str, Any]:
        """Get a comprehensive self-assessment of the AI system."""
        capabilities = self.assess_capabilities()
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'capabilities': capabilities,
            'interaction_metrics': {
                'total_interactions': self.interaction_count,
                'total_responses': self.performance_metrics['total_responses'],
                'total_improvements': self.performance_metrics['total_improvements']
            },
            'recent_insights': self.adaptation_history[-5:] if self.adaptation_history else []
        }
        
        return assessment

    def auto_improve(self, target_area: Optional[str] = None) -> Dict[str, Any]:
        """Automatically improve the AI system based on analysis."""
        self.logger.info(f"Starting auto-improvement {f'for {target_area}' if target_area else ''}")
        
        goal = target_area or 'enhance_overall_capabilities'
        improvement_result = self.self_improve(goal)
        
        improvement_result['timestamp'] = datetime.now().isoformat()
        self.performance_metrics['total_improvements'] += 1
        
        self.logger.info(f"Auto-improvement completed: {improvement_result['status']}")
        return improvement_result

    def learn_from_feedback(self, feedback: str, related_interaction_idx: Optional[int] = None) -> Dict[str, Any]:
        """Learn from user feedback to improve future responses."""
        self.logger.info(f"Learning from feedback: {feedback}")
        
        learning_result = {
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback,
            'status': 'processing',
            'actions_taken': []
        }
        
        # Feedback-driven improvement
        if 'too formal' in feedback.lower():
            learning_result['actions_taken'].append('Adjust formality level')
        if 'too technical' in feedback.lower():
            learning_result['actions_taken'].append('Simplify explanations')
        if 'more detail' in feedback.lower():
            learning_result['actions_taken'].append('Expand information')
        if 'less detail' in feedback.lower():
            learning_result['actions_taken'].append('Reduce verbosity')
        
        learning_result['status'] = 'completed'
        return learning_result

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        modification_count = len(self.adaptation_history)
        avg_interaction_length = (
            sum(len(h['response'].split()) for h in self.adaptation_history) / 
            max(modification_count, 1)
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy' if modification_count > 0 else 'initializing',
            'uptime_interactions': self.interaction_count,
            'modifications_applied': modification_count,
            'avg_response_length': avg_interaction_length,
            'learning_enabled': self.learning_enabled,
            'adaptive_ai_active': self.adaptive_ai is not None
        }

    def export_learning_report(self, output_path: str = 'logs/learning_report.json') -> None:
        """Export a comprehensive learning report."""
        import json
        
        report = {
            'export_timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health(),
            'self_assessment': self.get_self_assessment(),
            'interaction_history': self.adaptation_history,
            'performance_metrics': self.performance_metrics
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Learning report exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export learning report: {e}")

    def enable_learning(self) -> None:
        """Enable learning mode."""
        self.learning_enabled = True
        self.logger.info("Learning mode enabled")

    def disable_learning(self) -> None:
        """Disable learning mode."""
        self.learning_enabled = False
        self.logger.info("Learning mode disabled")

    def reset_learning(self) -> None:
        """Reset learning history and metrics."""
        self.interaction_count = 0
        self.adaptation_history.clear()
        self.performance_metrics = {
            'total_responses': 0,
            'total_improvements': 0,
            'avg_response_quality': 0.0
        }
        self.logger.info("Learning history reset")

    def get_capabilities_info(self) -> str:
        """Get human-readable capabilities information."""
        assessment = self.get_self_assessment()
        health = self.get_system_health()
        
        info = f"""
=== ADAPTIVE ASSISTANT CAPABILITIES ===
Status: {health['status']}
Uptime Interactions: {health['uptime_interactions']}
Learning Enabled: {health['learning_enabled']}
Modifications Applied: {health['modifications_applied']}

System Components:
- Conversation Memory: Active
- Knowledge Memory: {'Active' if self.knowledge.documents else 'Empty'}
- Self-Modification Engine: {'Active' if self.adaptive_ai else 'Inactive'}
- Voice System: {'Enabled' if self.config.use_voice else 'Disabled'}
- Language Model: Ready
- Tokenizer: Ready

Recent Insights:
{chr(10).join([f"  - {h['insights'][0]}" for h in assessment['recent_insights'][:3] if h.get('insights')])}

Performance:
- Total Responses: {health['uptime_interactions']}
- Avg Response Length: {health['avg_response_length']:.0f} words
- Improvements Proposed: {health['modifications_applied']}
================================
"""
        return info
