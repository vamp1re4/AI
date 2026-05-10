"""
Prompt utilities for building structured assistant conversations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


def clean_text(text: str) -> str:
    return ' '.join(text.strip().split())


@dataclass
class PromptTemplate:
    system_prompt: str = 'You are a helpful assistant.'
    user_prefix: str = 'User:'
    assistant_prefix: str = 'Assistant:'
    knowledge_prefix: str = 'Relevant knowledge:'
    max_history_messages: int = 10

    def build(self,
              history: List[Dict[str, str]],
              user_input: str,
              memory_text: Optional[str] = None) -> str:
        lines = [self.system_prompt]
        if memory_text:
            lines.append(f"{self.knowledge_prefix}\n{memory_text}")

        if history:
            lines.extend([f"{entry['source'].capitalize()}: {entry['text']}" for entry in history])

        lines.append(f"{self.user_prefix} {user_input}")
        lines.append(f"{self.assistant_prefix}")
        return clean_text('\n'.join(lines))
