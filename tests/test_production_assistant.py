import os

from src.core.memory import KnowledgeMemory
from src.core.production import ProductionAssistant, ProductionAssistantConfig
from src.core.text_processing import CharacterTokenizer


def test_knowledge_memory_retrieval():
    memory = KnowledgeMemory(capacity=5)
    memory.add_document('First', 'This is a test document about AI and optimization.')
    memory.add_document('Second', 'Another document describing training and learning.')
    results = memory.retrieve('training AI', top_k=2)

    assert len(results) == 2
    assert any('AI' in doc['text'] or 'training' in doc['text'] for doc in results)


def test_production_assistant_from_config_and_response():
    config_path = os.path.join('configs', 'assistant_config.yaml')
    config = ProductionAssistantConfig.load(config_path)

    tokenizer = CharacterTokenizer()
    tokenizer.fit(['hello world'])

    assistant = ProductionAssistant.from_config(tokenizer=tokenizer, config_path=config_path)
    assistant.knowledge.add_document('Tip', 'Keep responses short and relevant.')
    response = assistant.respond('What should I do next?')

    assert isinstance(response, str)
    assert response != ''
    assert assistant.memory.get_recent(1)[0]['source'] == 'assistant'
