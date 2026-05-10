"""
Memory systems for conversational AI and retrieval.

Provides lightweight memory buffers for conversation history,
recent context retrieval, and keyword-based retrieval.
"""

import time
from typing import List, Dict, Optional


class MemorySystem:
    """Simple memory buffer for storing text entries."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.entries: List[Dict[str, object]] = []

    def add(self, text: str, source: str = 'user') -> None:
        """Add a memory entry."""
        entry = {
            'source': source,
            'text': text,
            'timestamp': time.time(),
        }
        self.entries.append(entry)

        if len(self.entries) > self.capacity:
            self.entries.pop(0)

    def get_recent(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the most recent N memory entries."""
        return self.entries[-n:]

    def search(self, query: str) -> List[Dict[str, object]]:
        """Return entries containing the query string."""
        query_lower = query.lower()
        return [entry for entry in self.entries if query_lower in entry['text'].lower()]

    def summarize(self, max_entries: int = 5) -> str:
        """Return a simple summary of recent entries."""
        recent = self.get_recent(max_entries)
        return ' '.join([f"{entry['source']}: {entry['text']}" for entry in recent])


class ConversationMemory(MemorySystem):
    """Conversation memory for assistant context management."""

    def add_user(self, text: str) -> None:
        self.add(text, source='user')

    def add_assistant(self, text: str) -> None:
        self.add(text, source='assistant')

    def add_system(self, text: str) -> None:
        self.add(text, source='system')

    def get_context_tokens(self, tokenizer, max_tokens: int) -> List[int]:
        """Return the most recent conversation context as token ids."""
        context_tokens: List[int] = []

        for entry in reversed(self.entries):
            prefix = f"{entry['source'].capitalize()}: "
            chunk = tokenizer.encode(prefix + entry['text'] + '\n')
            if len(context_tokens) + len(chunk) > max_tokens:
                remaining = max_tokens - len(context_tokens)
                if remaining <= 0:
                    break
                chunk = chunk[-remaining:]
            context_tokens = chunk + context_tokens

        return context_tokens[-max_tokens:]

    def to_prompt(self, max_entries: Optional[int] = None) -> str:
        """Format recent conversation history into a prompt string."""
        recent_entries = self.get_recent(max_entries) if max_entries is not None else self.entries
        return '\n'.join(f"{entry['source'].capitalize()}: {entry['text']}" for entry in recent_entries)


class KnowledgeMemory(MemorySystem):
    """Document memory store with simple relevance retrieval."""

    def __init__(self, capacity: int = 200):
        super().__init__(capacity)
        self.documents: List[Dict[str, str]] = []

    def add_document(self, title: str, text: str) -> None:
        """Add a document snippet to the knowledge base."""
        document = {
            'title': title,
            'text': text,
            'timestamp': time.time(),
        }
        self.documents.append(document)
        if len(self.documents) > self.capacity:
            self.documents.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve the most relevant documents for a query."""
        query_terms = set(query.lower().split())

        def score_document(doc: Dict[str, str]) -> int:
            doc_terms = set(doc['text'].lower().split())
            return len(query_terms.intersection(doc_terms))

        sorted_docs = sorted(self.documents, key=score_document, reverse=True)
        return sorted_docs[:top_k]

    def retrieve_text(self, query: str, top_k: int = 3) -> str:
        """Return a combined text summary of the most relevant documents."""
        results = self.retrieve(query, top_k=top_k)
        return '\n'.join(f"{doc['title']}: {doc['text']}" for doc in results)
