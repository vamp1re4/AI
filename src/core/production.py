"""
Production-ready assistant wrapper.

Combines model training, checkpointing, conversation memory, prompt formatting,
voice output, and configurable behavior for a more realistic AI assistant workflow.
Includes self-modification capabilities for continuous adaptation and improvement.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import yaml

from .assistant import AssistantSystem
from .language_model import MiniLanguageModel
from .memory import ConversationMemory, KnowledgeMemory
from .prompt import PromptTemplate
from .voice import VoiceSystem
from .text_processing import Tokenizer
from .self_modifier import AdaptiveAI


@dataclass
class ProductionAssistantConfig:
    system_prompt: str = 'You are a production-ready AI assistant.'
    max_seq_len: int = 32
    max_history_messages: int = 10
    response_length: int = 20
    temperature: float = 0.9
    sample: bool = True
    checkpoint_dir: str = 'checkpoints/assistant'
    model_path: str = 'models/assistant_model.npz'
    history_path: str = 'logs/assistant_history.json'
    use_voice: bool = False
    enable_self_modification: bool = True
    optimizer: str = 'adam'
    optimizer_params: dict = field(default_factory=lambda: {'weight_decay': 1e-4})
    training_epochs: int = 20
    training_batch_size: int = 8
    training_val_split: float = 0.2
    training_patience: int = 5
    training_seq_length: int = 8

    @classmethod
    def load(cls, filepath: str) -> 'ProductionAssistantConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.__dict__, f)


class ProductionAssistant:
    def __init__(self,
                 tokenizer: Tokenizer,
                 language_model: MiniLanguageModel,
                 config: Optional[ProductionAssistantConfig] = None,
                 memory: Optional[ConversationMemory] = None,
                 knowledge: Optional[KnowledgeMemory] = None,
                 voice: Optional[VoiceSystem] = None):
        self.tokenizer = tokenizer
        self.model = language_model
        self.config = config or ProductionAssistantConfig()
        self.memory = memory or ConversationMemory()
        self.knowledge = knowledge or KnowledgeMemory()
        self.voice = voice or VoiceSystem()
        self.prompt_template = PromptTemplate(
            system_prompt=self.config.system_prompt,
            max_history_messages=self.config.max_history_messages
        )
        
        # Self-modification capability
        self.adaptive_ai = AdaptiveAI() if self.config.enable_self_modification else None

        if self.config.system_prompt:
            self.memory.add_system(self.config.system_prompt)

    @classmethod
    def from_config(cls, tokenizer: Tokenizer, config_path: str) -> 'ProductionAssistant':
        config = ProductionAssistantConfig.load(config_path)
        model = MiniLanguageModel(
            vocab_size=tokenizer.get_vocab_size(),
            max_seq_len=config.max_seq_len,
            optimizer=config.optimizer,
            optimizer_params=config.optimizer_params
        )
        return cls(tokenizer=tokenizer, language_model=model, config=config)

    def train(self, texts: List[str], verbose: bool = True):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        return self.model.train(
            texts=texts,
            tokenizer=self.tokenizer,
            seq_length=self.config.training_seq_length,
            epochs=self.config.training_epochs,
            batch_size=self.config.training_batch_size,
            val_split=self.config.training_val_split,
            patience=self.config.training_patience,
            verbose=verbose,
            checkpoint_dir=self.config.checkpoint_dir
        )

    def build_prompt(self, user_text: str) -> str:
        history = self.memory.get_recent(self.config.max_history_messages)
        memory_text = self.knowledge.retrieve_text(user_text, top_k=3) if self.knowledge.documents else ''
        return self.prompt_template.build(history=history, user_input=user_text, memory_text=memory_text)

    def respond(self, user_text: str) -> str:
        prompt_text = self.build_prompt(user_text)
        self.memory.add_user(user_text)

        token_ids = self.tokenizer.encode(prompt_text)
        token_ids = token_ids[-self.config.max_seq_len:]

        generated = self.model.generate(
            initial_tokens=token_ids,
            max_new_tokens=self.config.response_length,
            temperature=self.config.temperature,
            sample=self.config.sample
        )

        response_tokens = generated[len(token_ids):]
        response = self.tokenizer.decode(response_tokens.tolist())
        response = self._clean_response(response)
        self.memory.add_assistant(response)

        if self.config.use_voice:
            self.voice.speak(response)

        return response

    def _clean_response(self, text: str) -> str:
        for token in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
            text = text.replace(token, '')
        return ' '.join(text.split()).strip()

    def save_model(self) -> None:
        os.makedirs(os.path.dirname(self.config.model_path) or '.', exist_ok=True)
        self.model.save(self.config.model_path)

    def load_model(self) -> None:
        self.model.load(self.config.model_path)

    def save_history(self) -> None:
        os.makedirs(os.path.dirname(self.config.history_path) or '.', exist_ok=True)
        with open(self.config.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory.entries, f, indent=2)

    def load_history(self) -> None:
        if not os.path.exists(self.config.history_path):
            return
        with open(self.config.history_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        self.memory.entries = entries

    def assess_capabilities(self) -> Dict[str, Any]:
        """Assess current AI capabilities and suggest improvements."""
        if not self.config.enable_self_modification or not self.adaptive_ai:
            return {}
        return self.adaptive_ai.assess_current_state()

    def plan_adaptation(self, goal: str) -> List[Dict[str, str]]:
        """Plan adaptations to achieve a specific goal."""
        if not self.config.enable_self_modification or not self.adaptive_ai:
            return []
        return self.adaptive_ai.plan_adaptation(goal)

    def propose_enhancement(self, enhancement_type: str, **kwargs) -> Dict[str, Any]:
        """Propose an enhancement to the AI system."""
        if not self.config.enable_self_modification or not self.adaptive_ai:
            return {}
        return self.adaptive_ai.propose_enhancement(enhancement_type, **kwargs)

    def self_improve(self, goal: str) -> Dict[str, Any]:
        """Attempt to improve the AI system based on a goal."""
        if not self.config.enable_self_modification or not self.adaptive_ai:
            return {'status': 'disabled', 'message': 'Self-modification is disabled'}
        
        # Plan adaptation
        plan = self.plan_adaptation(goal)
        
        # Execute adaptation
        results = self.adaptive_ai.execute_adaptation(plan)
        results['goal'] = goal
        
        return results

    def save_modification_log(self, output_path: str = 'logs/modification_history.json') -> None:
        """Save the history of all modifications."""
        if self.config.enable_self_modification and self.adaptive_ai:
            self.adaptive_ai.executor.save_modification_log(output_path)
