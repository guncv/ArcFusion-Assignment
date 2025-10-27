"""
Document Processor - Handles document ingestion, chunking, and vector storage.

Processes documents through the complete RAG pipeline:
1. Load documents using reader
2. Chunk documents using semantic chunker
3. Store chunks in vector database
"""

from pathlib import Path
from typing import List
from llama_index.core.schema import Document as LlamaDocument, TextNode

from src.infras.chunking.factory import ChunkerFactory
from src.infras.vector_db.factory import VectorStoreFactory
from src.infras.ingestion.factory import ReaderFactory


class DocumentProcessor:
    """
    Document processor for RAG pipeline.

    Handles the complete document processing workflow from ingestion to vector storage.
    """

    def __init__(self):
        """Initialize document processor with factories."""
        self.chunker = ChunkerFactory.get()
        self.reader = ReaderFactory.get()
        self.vector_store = VectorStoreFactory.get()

    def extract_text_from_pdf(self, pdf_path: str) -> List[LlamaDocument]:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of LlamaIndex Document objects with metadata

        Raises:
            ValueError: If PDF cannot be extracted
        """
        try:
            documents = self.reader.load_data(pdf_path)
            return documents

        except Exception as e:
            raise ValueError(f"Failed to extract PDF: {str(e)}")

    def chunk_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        """
        Chunk documents into smaller text nodes.

        Args:
            documents: List of LlamaIndex Document objects

        Returns:
            List of TextNode objects (chunks)

        Raises:
            Exception: If chunking fails
        """
        try:
            chunked_nodes = self.chunker.split_documents(documents)
            return chunked_nodes

        except Exception as e:
            raise

    def process_pdf(self, pdf_path: str) -> List[TextNode]:
        """
        Process a single PDF file through the complete pipeline.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of TextNode chunks

        Raises:
            ValueError: If processing fails
        """
        documents = self.extract_text_from_pdf(pdf_path)
        chunked_nodes = self.chunk_documents(documents)
        return chunked_nodes

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[TextNode]:
        """
        Process multiple PDF files and store in vector database.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of all TextNode chunks from all PDFs

        Note:
            Failed PDFs are skipped silently to allow batch processing to continue.
        """
        all_chunks: List[TextNode] = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                # Skip failed PDFs and continue processing
                continue

        # Store all chunks in vector database
        self.vector_store.add_documents(all_chunks)

        return all_chunks
