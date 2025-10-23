import fitz
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.log.logger import logger
from core.config.config import nested_config as config

class DocumentProcessor:
    
    def __init__( self):
        self.chunk_size = config["rag"]["chunking"]["chunk_size"]
        self.chunk_overlap = config["rag"]["chunking"]["chunk_overlap"]
        self.separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            ", ",
            " ",
            ""
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            keep_separator=True,
        )

    def extract_text_from_pdf(self, pdf_path: str, extract_metadata: bool = True) -> List[Document]:
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            documents = []

            logger.info(f"[DocumentProcessor] Processing PDF: {pdf_file.name} ({len(doc)} pages)")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if not text.strip():
                    logger.debug(f"[DocumentProcessor] Skipping empty page {page_num + 1}")
                    continue

                metadata = {
                    "source": str(pdf_file),
                    "page": page_num + 1,
                    "total_pages": len(doc),
                }

                if extract_metadata:
                    metadata.update({
                        "file_name": pdf_file.name,
                        "file_size": pdf_file.stat().st_size,
                    })

                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))

            doc.close()
            logger.info(f"[DocumentProcessor] Extracted {len(documents)} pages from {pdf_file.name}")

            return documents

        except Exception as e:
            logger.error(f"[DocumentProcessor] Error extracting PDF {pdf_path}: {e}")
            raise ValueError(f"Failed to extract PDF: {str(e)}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            chunked_docs = self.text_splitter.split_documents(documents)

            for i, chunk in enumerate(chunked_docs):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)

            logger.info(
                f"[DocumentProcessor] Split {len(documents)} documents into "
                f"{len(chunked_docs)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})"
            )

            return chunked_docs

        except Exception as e:
            logger.error(f"[DocumentProcessor] Error chunking documents: {e}")
            raise

    def process_pdf(self, pdf_path: str, extract_metadata: bool = True) -> List[Document]:
        documents = self.extract_text_from_pdf(pdf_path, extract_metadata)
        chunked_docs = self.chunk_documents(documents)

        return chunked_docs

    def process_multiple_pdfs(self, pdf_paths: List[str], extract_metadata: bool = True ) -> List[Document]:
        all_chunks = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path, extract_metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"[DocumentProcessor] Failed to process {pdf_path}: {e}")
                continue

        logger.info(
            f"[DocumentProcessor] Processed {len(pdf_paths)} PDFs → "
            f"{len(all_chunks)} total chunks"
        )

        return all_chunks
