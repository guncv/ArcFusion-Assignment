#!/usr/bin/env python3
"""
Auto-Ingestion Module

Automatically ingests PDF documents from a configured directory when the application starts.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional
from core.log.logger import logger
from infrastructure.rag import get_rag_pipeline
from core.config.config import nested_config as config


class AutoIngestionManager:
    """Manages automatic document ingestion on application startup."""
    
    def __init__(self):
        self.config = config["rag"]["auto_ingestion"]
        self.enabled = self.config.get("enabled", False)
        self.documents_directory = self.config.get("documents_directory", "./documents")
        self.clear_existing = self.config.get("clear_existing", False)
        self.on_startup = self.config.get("on_startup", True)
        
    def should_auto_ingest(self) -> bool:
        """Check if auto-ingestion should be performed."""
        return self.enabled and self.on_startup
    
    def get_pdf_files(self) -> List[Path]:
        """Get all PDF files from the configured documents directory."""
        docs_dir = Path(self.documents_directory)
        
        if not docs_dir.exists():
            logger.warning(f"[AutoIngestion] Documents directory not found: {self.documents_directory}")
            return []
        
        if not docs_dir.is_dir():
            logger.warning(f"[AutoIngestion] Path is not a directory: {self.documents_directory}")
            return []
        
        # Find all PDF files recursively
        pdf_files = list(docs_dir.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.info(f"[AutoIngestion] No PDF files found in {self.documents_directory}")
            return []
        
        logger.info(f"[AutoIngestion] Found {len(pdf_files)} PDF files to ingest")
        return pdf_files
    
    async def ingest_documents(self) -> Optional[dict]:
        """Perform automatic document ingestion."""
        if not self.should_auto_ingest():
            logger.info("[AutoIngestion] Auto-ingestion disabled or not configured for startup")
            return None
        
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.info("[AutoIngestion] No PDF files to ingest")
            return None
        
        try:
            logger.info(f"[AutoIngestion] Starting auto-ingestion of {len(pdf_files)} PDF files")

            pdf_paths = [str(p) for p in pdf_files]

            pipeline = get_rag_pipeline()
            result = pipeline.ingest_documents(
                pdf_paths=pdf_paths,
                clear_existing=self.clear_existing
            )
            
            if result.get("status") == "success":
                logger.info(
                    f"[AutoIngestion] ✓ Auto-ingestion successful! "
                    f"PDFs: {result.get('pdfs_processed', 0)}, "
                    f"Chunks: {result.get('chunks_created', 0)}, "
                    f"Indexed: {result.get('documents_indexed', 0)}"
                )
            else:
                logger.error(f"[AutoIngestion] ✗ Auto-ingestion failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"[AutoIngestion] Error during auto-ingestion: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
    def log_configuration(self):
        """Log the current auto-ingestion configuration."""
        logger.info(f"[AutoIngestion] Configuration:")
        logger.info(f"  - Enabled: {self.enabled}")
        logger.info(f"  - Documents Directory: {self.documents_directory}")
        logger.info(f"  - Clear Existing: {self.clear_existing}")
        logger.info(f"  - On Startup: {self.on_startup}")


# Global instance
auto_ingestion_manager = AutoIngestionManager()


async def perform_auto_ingestion() -> Optional[dict]:
    """
    Perform automatic document ingestion if configured.
    
    Returns:
        dict: Ingestion result or None if not performed
    """
    auto_ingestion_manager.log_configuration()
    return await auto_ingestion_manager.ingest_documents()


def get_auto_ingestion_status() -> dict:
    """Get the current status of auto-ingestion configuration."""
    return {
        "enabled": auto_ingestion_manager.enabled,
        "documents_directory": auto_ingestion_manager.documents_directory,
        "clear_existing": auto_ingestion_manager.clear_existing,
        "on_startup": auto_ingestion_manager.on_startup,
        "should_auto_ingest": auto_ingestion_manager.should_auto_ingest(),
        "pdf_files_found": len(auto_ingestion_manager.get_pdf_files())
    }
