from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode

from infras.rag.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking strategy with configurable chunk size and overlap.
    Simple and fast chunking approach.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._chunker = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        """
        Split documents into fixed-size chunks.

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
                    node.metadata["chunking_strategy"] = "fixed_size"
                    all_nodes.append(node)

            except Exception as e:
                # Fallback: create a single node with the entire document
                fallback_node = TextNode(
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": 0,
                        "chunk_size": len(doc.text),
                        "chunking_strategy": "fixed_size",
                        "error": str(e)
                    }
                )
                all_nodes.append(fallback_node)

        return all_nodes

    def split_text(self, text: str, metadata: dict = None) -> List[TextNode]:
        """
        Split a single text into fixed-size chunks.

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
        return "fixed_size"
