from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode
from src.infras.chunking.base import BaseChunker
from src.infras.embedding.base import BaseEmbedding
from src.infras.log import logger as log
from src.config import config

class RecursiveChunker(BaseChunker):
    
    def __init__(self, embedding_provider: BaseEmbedding):
        self._embedding_provider = embedding_provider

        # Get config or use defaults
        rag_config = config.get("rag", {})
        chunking_config = rag_config.get("chunking", {})

        # Configurable parameters
        self._chunk_size = chunking_config.get("chunk_size", 1024)
        self._chunk_overlap = chunking_config.get("chunk_overlap", 200)

        self._chunker = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        all_nodes: List[TextNode] = []

        for doc in documents:
            try:
                nodes = self._chunker.get_nodes_from_documents([doc])

                for i, node in enumerate(nodes):
                    node.metadata["chunk_id"] = i
                    node.metadata["chunk_size"] = len(node.text)
                    node.metadata["chunking_strategy"] = "recursive"
                    all_nodes.append(node)

            except Exception as e:
                fallback_node = TextNode(
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": 0,
                        "chunk_size": len(doc.text),
                        "chunking_strategy": "recursive_fallback",
                        "error": str(e)
                    }
                )
                all_nodes.append(fallback_node)
                
        return all_nodes

    def split_text(self, text: str, metadata: dict = None) -> List[TextNode]:
        if metadata is None:
            metadata = {}

        doc = LlamaDocument(text=text, metadata=metadata)
        return self.split_documents([doc])

    @property
    def name(self) -> str:
        return "recursive"
