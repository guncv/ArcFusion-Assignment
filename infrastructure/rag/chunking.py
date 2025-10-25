from typing import List
from core.config.config import nested_config as config
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document as LlamaDocument, TextNode

class ChunkingManager:

    def __init__(self) -> None:
        self.buffer_size = config["rag"]["chunking"]["llamaindex_buffer_size"]
        self.breakpoint_percentile_threshold = config["rag"]["chunking"]["llamaindex_breakpoint_threshold"]

        embedding_model_name = config["rag"]["embedding_model"]
        api_key = config["rag"]["embedding_api_key"]
        self.embed_model = OpenAIEmbedding(
            model_name=embedding_model_name,
            api_key=api_key,
        )

        self.chunker = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            embed_model=self.embed_model
        )

    def split_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        all_nodes: List[TextNode] = []

        for doc in documents:
            try:
                nodes = self.chunker.get_nodes_from_documents([doc])

                for i, node in enumerate(nodes):
                    node.metadata["chunk_id"] = i
                    node.metadata["chunk_size"] = len(node.text)
                    all_nodes.append(node)

            except Exception as e:
                fallback_node = TextNode(
                    text=doc.text,
                    metadata={**doc.metadata, "chunk_id": 0, "chunk_size": len(doc.text)}
                )
                all_nodes.append(fallback_node)

        return all_nodes
