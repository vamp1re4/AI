import os

import numpy as np

from src.core.assistant import AssistantSystem
from src.core.language_model import MiniLanguageModel
from src.core.memory import ConversationMemory, MemorySystem
from src.core.text_processing import CharacterTokenizer
from src.core.voice import VoiceSystem


def test_memory_system_add_and_search():
    memory = MemorySystem(capacity=3)
    memory.add('hello there', source='user')
    memory.add('assistant reply', source='assistant')
    memory.add('system note', source='system')

    assert len(memory.entries) == 3
    assert memory.search('assistant')[0]['source'] == 'assistant'
    assert 'hello there' in memory.summarize()


def test_assistant_system_conversation_flow():
    sample_texts = ['hello world', 'this is a test']
    tokenizer = CharacterTokenizer()
    tokenizer.fit(sample_texts)

    model = MiniLanguageModel(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_seq_len=16,
        learning_rate=0.01,
        optimizer='adam'
    )

    assistant = AssistantSystem(
        model=model,
        tokenizer=tokenizer,
        memory=ConversationMemory(capacity=10),
        seq_length=16,
        system_prompt='You are an assistant.'
    )

    response = assistant.respond('hello', max_new_tokens=5, temperature=1.0)
    assert isinstance(response, str)
    assert any(entry['source'] == 'assistant' for entry in assistant.get_history())


def test_voice_system_fallback(tmp_path):
    voice = VoiceSystem()
    output_file = tmp_path / 'voice_output.txt'
    result_path = voice.synthesize('test message', output_path=str(output_file))

    assert result_path == str(output_file)
    assert os.path.exists(result_path)
    with open(result_path, 'r', encoding='utf-8') as f:
        assert 'test message' in f.read()
