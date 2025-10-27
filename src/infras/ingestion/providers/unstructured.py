"""
Unstructured Reader Provider - Wrapper for LlamaIndex's UnstructuredReader.

Provides document ingestion using the Unstructured library for parsing
PDFs, DOCX, HTML, and other document formats.
"""

from typing import List
from pathlib import Path
from llama_index.readers.file import UnstructuredReader
from llama_index.core.schema import Document as LlamaDocument

from src.infras.ingestion.base import BaseReader


class UnstructuredReaderProvider(BaseReader):
    """
    Unstructured reader implementation using LlamaIndex's UnstructuredReader.

    Supports multiple document formats including PDF, DOCX, HTML, and more.
    """

    def __init__(self):
        """Initialize the Unstructured reader."""
        self._reader = UnstructuredReader()

    def load_data(self, file_path: str, **kwargs) -> List[LlamaDocument]:
        """
        Load and parse a document file using Unstructured.

        Args:
            file_path: Path to the document file
            **kwargs: Additional parameters (unused for Unstructured)

        Returns:
            List of LlamaIndex Document objects with metadata

        Raises:
            ValueError: If file cannot be read or parsed
        """
        try:
            pdf_file = Path(file_path)

            if not pdf_file.exists():
                raise ValueError(f"File not found: {file_path}")

            # Load documents using UnstructuredReader
            documents = self._reader.load_data(file=file_path)

            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "source": str(pdf_file),
                    "file_name": pdf_file.name,
                    "file_size": pdf_file.stat().st_size,
                    "reader": self.reader_name
                })

            return documents

        except Exception as e:
            raise ValueError(f"Failed to load document with Unstructured: {str(e)}")

    @property
    def reader_name(self) -> str:
        """
        Get the name of the reader.

        Returns:
            Reader name string
        """
        return "unstructured"
