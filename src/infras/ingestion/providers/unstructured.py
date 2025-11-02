from typing import List
from pathlib import Path
from llama_index.core.schema import Document as LlamaDocument
from src.infras.ingestion.base import BaseReader
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_text
from src.infras.log import logger as log

class UnstructuredReaderProvider(BaseReader):
    def __init__(self):
        self.strategy = "hi_res"  # use ML-based layout parsing

    def load_data(self, file_path: str, **kwargs) -> List[LlamaDocument]:
        try:
            pdf_file = Path(file_path)
            if not pdf_file.exists():
                raise ValueError(f"File not found: {file_path}")

            elements = partition_pdf(
                filename=str(pdf_file),
                strategy=self.strategy,
                extract_images_in_pdf=False,   # remove images
                infer_table_structure=False,   # remove tables
                languages=["eng"],
            )
    
            markdown_text = convert_to_text(elements)

            document = LlamaDocument(
                text=markdown_text,
                metadata={
                    "source": str(pdf_file),
                    "file_name": pdf_file.name,
                    "file_size": pdf_file.stat().st_size,
                    "reader": self.reader_name,
                },
            )

            return [document]

        except Exception as e:
            raise ValueError(f"Failed to load document with Unstructured Markdown: {str(e)}")

    @property
    def reader_name(self) -> str:
        return "unstructured_markdown"
