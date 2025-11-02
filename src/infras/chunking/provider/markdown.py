from typing import List
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document as LlamaDocument, TextNode
from src.infras.chunking.base import BaseChunker
from src.infras.embedding.base import BaseEmbedding
from src.infras.log import logger as log
class MarkdownChunker(BaseChunker):

    def __init__(self, embedding_provider: BaseEmbedding):
        self._embedding_provider = embedding_provider
        self._chunker = MarkdownNodeParser()

    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        all_nodes: List[TextNode] = []

        for doc in documents:
            try:
                nodes = self._chunker.get_nodes_from_documents([doc])

                for i, node in enumerate(nodes):
                    node.metadata["chunk_id"] = i
                    node.metadata["chunk_size"] = len(node.text)
                    node.metadata["chunking_strategy"] = "markdown"
                    all_nodes.append(node)

            except Exception as e:
                fallback_node = TextNode(
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": 0,
                        "chunk_size": len(doc.text),
                        "chunking_strategy": "markdown_fallback",
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
        return "markdown"
