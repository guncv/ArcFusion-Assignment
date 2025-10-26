from typing import Optional, List
from pathlib import Path
from core.config.config import nested_config as config
from infrastructure.rag.document_processor import DocumentProcessor

class AutoIngestionManager:

    def __init__(self):
        self.config = config["rag"]["auto_ingestion"]
        self.enabled = self.config.get("enabled", False)
        self.documents_directory = self.config.get("documents_directory", ["./documents"])
        self.clear_existing = self.config.get("clear_existing", False)
        self.on_startup = self.config.get("on_startup", True)
        self.document_processor = DocumentProcessor()

    def _scan_directories_for_pdfs(self, directories: List[str]) -> List[str]:
        pdf_files = []

        for directory in directories:
            dir_path = Path(directory)

            if not dir_path.exists():
                continue

            if dir_path.is_file():
                if dir_path.suffix.lower() == '.pdf':
                    pdf_files.append(str(dir_path))
            elif dir_path.is_dir():
                pdfs_in_dir = list(dir_path.glob("**/*.pdf"))
                pdf_files.extend([str(pdf) for pdf in pdfs_in_dir])

        return pdf_files

    async def ingest_documents(self) -> Optional[dict]:
        try:
            pdf_files = self._scan_directories_for_pdfs(self.documents_directory)

            if not pdf_files:
                return {"status": "success", "chunks": [], "message": "No PDF files found"}

            chunks = self.document_processor.process_multiple_pdfs(pdf_files)

            return {"status": "success", "chunks": chunks, "files_processed": len(pdf_files)}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
        
ingestion_manager = AutoIngestionManager()