from typing import List, Any
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaOpenAIEmbedding
from src.config import config
from src.infras.embedding.base import BaseEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding as LlamaBaseEmbedding

class OpenAIEmbeddingProvider(BaseEmbedding):

    def __init__(self):
        self._model_name = config.get("embedding", {}).get("openai", {}).get("model_name")
        self._api_key = config.get("embedding", {}).get("openai", {}).get("api_key")
        
        if not self._model_name or not self._api_key:
            raise ValueError("Model name and API key are required")

        self._embed_model = LlamaOpenAIEmbedding(
            model_name=self._model_name,
            api_key=self._api_key,
        )

    def get_text_embedding(self, text: str) -> List[float]:
        return self._embed_model.get_text_embedding(text)

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_model.get_text_embedding_batch(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        return self._embed_model.get_query_embedding(query)
    
    def get_text_embedding_model(self) -> LlamaBaseEmbedding:
        return self._embed_model
