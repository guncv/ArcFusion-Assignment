from typing import List
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument, TextNode
from src.infras.chunking.base import BaseChunker
from src.infras.embedding.base import BaseEmbedding

class SemanticNodeChunker(BaseChunker):
    def __init__(self,embedding_provider: BaseEmbedding):
        self._embedding_provider = embedding_provider
        self._buffer_size = 1
        self._breakpoint_percentile_threshold = 85

        embed_model = embedding_provider.get_text_embedding_model()

        self._chunker = SemanticSplitterNodeParser(
            buffer_size=self._buffer_size,
            breakpoint_percentile_threshold=self._breakpoint_percentile_threshold,
            embed_model=embed_model
        )

    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        all_nodes: List[TextNode] = []

        for doc in documents:
            try:
                nodes = self._chunker.get_nodes_from_documents([doc])

                for i, node in enumerate(nodes):
                    node.metadata["chunk_id"] = i
                    node.metadata["chunk_size"] = len(node.text)
                    node.metadata["chunking_strategy"] = "semantic"
                    all_nodes.append(node)

            except Exception as e:
                # Fallback: create a single node with the entire document
                fallback_node = TextNode(
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": 0,
                        "chunk_size": len(doc.text),
                        "chunking_strategy": "semantic",
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
        return "semantic_node"
    
