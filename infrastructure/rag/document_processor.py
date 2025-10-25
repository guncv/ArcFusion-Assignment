from pathlib import Path
from typing import List
from llama_index.readers.file import UnstructuredReader
from llama_index.core.schema import Document as LlamaDocument, TextNode
from infrastructure.rag.chunking import ChunkingManager
from infrastructure.vector_db.vector_store import VectorStoreManager

class DocumentProcessor:
    def __init__(self):
        self.chunker = ChunkingManager()
        self.reader = UnstructuredReader()
        self.vector_store = VectorStoreManager()

    def extract_text_from_pdf(self, pdf_path: str) -> List[LlamaDocument]:
        pdf_file = Path(pdf_path)

        try:
            documents = self.reader.load_data(file=pdf_path)
            for doc in documents:
                doc.metadata.update({
                    "source": str(pdf_file),
                    "file_name": pdf_file.name,
                    "file_size": pdf_file.stat().st_size,
                })

            return documents

        except Exception as e:
            raise ValueError(f"Failed to extract PDF: {str(e)}")

    def chunk_documents(self, documents: List[LlamaDocument]) -> List[TextNode]:
        try:
            chunked_nodes = self.chunker.split_documents(documents)
            return chunked_nodes

        except Exception as e:
            raise

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
                continue
        
        self.vector_store.add_documents(all_chunks)

        return all_chunks