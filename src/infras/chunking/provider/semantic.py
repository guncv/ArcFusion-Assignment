from typing import List
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument, TextNode

from infras.rag.chunking.base import BaseChunker
from infras.embedding.base import BaseEmbedding


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy using LlamaIndex's SemanticSplitterNodeParser.
    Splits text based on semantic similarity between sentences.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbedding,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
    ):
        """
        Initialize semantic chunker.

        Args:
            embedding_provider: Embedding provider for semantic similarity
            buffer_size: Number of sentences to group together
            breakpoint_percentile_threshold: Percentile threshold for sentence splits
        """
        self._embedding_provider = embedding_provider
        self._buffer_size = buffer_size
        self._breakpoint_percentile_threshold = breakpoint_percentile_threshold

        # Get the underlying LlamaIndex embed model
        embed_model = embedding_provider.llama_embed_model

        self._chunker = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model
        )

    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        """
        Split documents into chunks using semantic similarity.

        Args:
            documents: List of LlamaIndex Document objects to chunk

        Returns:
            List of TextNode objects (chunks) with metadata
        """
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
        """
        Split a single text into chunks using semantic similarity.

        Args:
            text: Text string to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextNode objects (chunks)
        """
        if metadata is None:
            metadata = {}

        doc = LlamaDocument(text=text, metadata=metadata)
        return self.split_documents([doc])

    @property
    def strategy_name(self) -> str:
        """
        Get the name of the chunking strategy.

        Returns:
            Strategy name string
        """
        return "semantic"
