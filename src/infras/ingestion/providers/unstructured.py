from typing import List
from pathlib import Path
from llama_index.readers.file import UnstructuredReader
from llama_index.core.schema import Document as LlamaDocument
from src.infras.ingestion.base import BaseReader

class UnstructuredReaderProvider(BaseReader):
    def __init__(self):
        self._reader = UnstructuredReader()

    def load_data(self, file_path: str, **kwargs) -> List[LlamaDocument]:
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
        return "unstructured"
