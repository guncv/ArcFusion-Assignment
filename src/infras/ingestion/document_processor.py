from typing import List
from llama_index.core.schema import Document as LlamaDocument, TextNode

from src.infras.chunking.factory import ChunkerFactory
from src.infras.vector_db.factory import VectorStoreFactory
from src.infras.ingestion.factory import ReaderFactory


class DocumentProcessor:
    def __init__(self):
        self.chunker = ChunkerFactory.get()
        self.reader = ReaderFactory.get()
        self.vector_store = VectorStoreFactory.get()

    def extract_text_from_pdf(self, pdf_path: str) -> List[LlamaDocument]:
        try:
            documents = self.reader.load_data(pdf_path)
            return documents

        except Exception as e:
            raise ValueError(f"Failed to extract PDF: {str(e)}")

    def chunk_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        try:
            chunked_nodes = self.chunker.split_documents(documents)
            return chunked_nodes

        except Exception as e:
            raise ValueError(f"Failed to chunk documents: {str(e)}")

    def process_pdf(self, pdf_path: str) -> List[TextNode]:
        documents = self.extract_text_from_pdf(pdf_path)
        chunked_nodes = self.chunk_documents(documents)
        return chunked_nodes

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[TextNode]:
        all_chunks: List[TextNode] = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                raise ValueError(f"Failed to process PDF: {str(e)}")

        self.vector_store.add_documents(all_chunks)

        return all_chunks
