"""
Assistant system built on a mini language model and conversation memory.

Provides a lightweight conversational interface for prompts, responses,
and history-aware generation.
"""

import numpy as np
import numpy as np
from typing import List, Optional

from .language_model import MiniLanguageModel
from .memory import ConversationMemory
from .text_processing import Tokenizer


class AssistantSystem:
    """Simple assistant interface wrapping a language model and memory."""

    def __init__(self,
                 model: MiniLanguageModel,
                 tokenizer: Tokenizer,
                 memory: Optional[ConversationMemory] = None,
                 seq_length: int = 32,
                 system_prompt: str = 'You are a helpful assistant.'):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory or ConversationMemory()
        self.seq_length = seq_length
        self.system_prompt = system_prompt

        if system_prompt:
            self.memory.add_system(system_prompt)

    def add_user_message(self, text: str) -> None:
        self.memory.add_user(text)

    def add_assistant_message(self, text: str) -> None:
        self.memory.add_assistant(text)

    def respond(self, user_text: str, max_new_tokens: int = 20,
                temperature: float = 1.0, sample: bool = True) -> str:
        """Add a user message, generate an assistant response, and store it."""
        self.add_user_message(user_text)

        context_tokens = self.memory.get_context_tokens(self.tokenizer, self.seq_length)
        if len(context_tokens) == 0:
            context_tokens = self.tokenizer.encode(user_text)

        generated = self.model.model.generate(
            initial_tokens=np.array(context_tokens, dtype=int),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            sample=sample
        )

        response_tokens = generated[len(context_tokens):]
        response_text = self._clean_response(self.tokenizer.decode(response_tokens.tolist()))
        self.add_assistant_message(response_text)
        return response_text

    def _clean_response(self, text: str) -> str:
        """Clean decoded response text and remove special tokens."""
        for token in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
            text = text.replace(token, '')
        return ' '.join(text.split()).strip()

    def get_history(self) -> List[dict]:
        """Return the full conversation history."""
        return self.memory.entries
