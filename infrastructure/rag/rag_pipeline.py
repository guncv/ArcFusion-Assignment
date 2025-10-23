from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from infrastructure.rag.document_processor import DocumentProcessor
from infrastructure.vector_db.vector_store import VectorStoreManager
from .retrievers import HybridRetrieverManager
from infrastructure.rag.web_search import TavilyWebSearch
from domain.enums.llm_type import LLMType
from core.log.logger import logger
from core.config.config import nested_config as config

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to a knowledge base and the web.

Your task is to answer questions using the provided context. Follow these guidelines:

## Response Guidelines:
1. **Answer from context first**: Use the retrieved documents as your primary source
2. **Cite your sources**: Reference specific documents or web pages when possible
3. **Be honest about limitations**: If the context doesn't contain the answer, say so
4. **Use web results when needed**: If local context is insufficient, use web search results
5. **Be concise but complete**: Provide thorough answers without unnecessary verbosity

## Context Priority:
1. Retrieved documents from knowledge base (most reliable)
2. Web search results (for recent/external information)
3. General knowledge (only if context is insufficient)

## Citation Format:
When citing, use: [Source: filename.pdf, Page X] or [Source: URL]

Context from knowledge base:
{context}

{web_context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

class RAGPipeline:
    def __init__(self):
        self.collection_name = config["vector_db"]["collection_name"]
        self.top_k = config["rag"]["retrieval"]["top_k"]
        self.use_web_search_flag = config["rag"]["web_search"].get("enabled", True)

        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        self.web_search = TavilyWebSearch() if self.use_web_search_flag else None

        self.hybrid_retriever: Optional[HybridRetrieverManager] = None

        # Lazy initialization to avoid circular imports
        self._llm = None
        self._chain = None

    @property
    def llm(self):
        """Lazy load LLM to avoid circular imports."""
        if self._llm is None:
            from infrastructure.llm.loader import loadLLM
            self._llm = loadLLM(LLMType.REFINED_QUERY_AGENT)
        return self._llm

    @property
    def chain(self):
        """Lazy load chain to avoid circular imports."""
        if self._chain is None:
            self._chain = RAG_PROMPT | self.llm | StrOutputParser()
        return self._chain

    def ingest_documents(self, pdf_paths: List[str], clear_existing: bool = False) -> Dict[str, Any]:
        try:
            if clear_existing:
                self.vector_store.delete_collection()
                self.vector_store = VectorStoreManager()

            chunks = self.doc_processor.process_multiple_pdfs(pdf_paths)

            if not chunks:
                return {"status": "failed", "reason": "No content extracted"}

            doc_ids = self.vector_store.add_documents(chunks)
            self.hybrid_retriever = HybridRetrieverManager(documents=chunks)

            stats = {
                "status": "success",
                "pdfs_processed": len(pdf_paths),
                "chunks_created": len(chunks),
                "documents_indexed": len(doc_ids),
                "collection_name": self.collection_name,
            }

            logger.info(f"[RAGPipeline] Ingestion complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"[RAGPipeline] Error during ingestion: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def retrieve(self, query: str, use_web_search: Optional[bool] = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query using hybrid retrieval.

        Args:
            query: User query string
            use_web_search: Override web search setting

        Returns:
            Dict with local_docs and web_docs
        """
        try:
            local_docs = []
            web_docs = []

            # Retrieve local documents
            if not self.hybrid_retriever:
                logger.warning("[RAGPipeline] Hybrid retriever not initialized, creating new one")
                # Try to get documents from vector store
                from .retrievers import HybridRetrieverManager
                self.hybrid_retriever = HybridRetrieverManager()

            local_docs = self.hybrid_retriever.retrieve(query)

            # Retrieve web documents if enabled
            should_use_web = use_web_search if use_web_search is not None else self.use_web_search_flag
            if should_use_web and self.web_search:
                try:
                    web_docs = self.web_search.search_as_documents(query)
                except Exception as e:
                    logger.warning(f"[RAGPipeline] Web search failed: {e}")
                    web_docs = []

            return {
                "local_docs": local_docs,
                "web_docs": web_docs
            }

        except Exception as e:
            logger.error(f"[RAGPipeline] Error during retrieval: {e}", exc_info=True)
            return {"local_docs": [], "web_docs": []}

    async def generate(
        self,
        query: str,
        session_id: str,
        use_web_search: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved context.

        Args:
            query: User query
            session_id: Session ID for chat history
            use_web_search: Override web search setting

        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"[RAGPipeline] Generating answer for: '{query[:100]}...'")

        try:
            # Step 1: Retrieve relevant documents
            retrieval_results = self.retrieve(query, use_web_search)

            local_docs = retrieval_results.get("local_docs", [])
            web_docs = retrieval_results.get("web_docs", [])

            # Step 2: Format context
            context = self._format_context(local_docs)
            web_context = self._format_web_context(web_docs)

            # Step 3: Get chat history
            from infrastructure.llm.loader import getChatHistory
            chat_history = getChatHistory(session_id)

            # Step 4: Generate answer
            answer = await self.chain.ainvoke({
                "question": query,
                "context": context,
                "web_context": web_context,
                "history": chat_history.messages
            })

            # Step 5: Extract sources
            sources = self._extract_sources(local_docs, web_docs)

            result = {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
                "used_web_search": bool(web_docs),
                "query": query,
            }

            logger.info(f"[RAGPipeline] Generated answer with {len(sources)} sources")
            return result

        except Exception as e:
            logger.error(f"[RAGPipeline] Error during generation: {e}", exc_info=True)
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "error": str(e),
                "sources": [],
                "num_sources": 0,
            }

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context string."""
        if not documents:
            return "No relevant documents found in knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content[:500]  # Limit length

            context_parts.append(
                f"[Document {i}] (Source: {Path(source).name}, Page {page})\n{content}\n"
            )

        return "\n".join(context_parts)

    def _format_web_context(self, documents: List[Document]) -> str:
        """Format web search results as context string."""
        if not documents:
            return ""

        context_parts = ["Web Search Results:"]
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Untitled")
            url = doc.metadata.get("url", "")
            content = doc.page_content[:300]

            context_parts.append(
                f"[Web Result {i}] {title}\nURL: {url}\n{content}\n"
            )

        return "\n".join(context_parts)

    def _extract_sources(
        self,
        local_docs: List[Document],
        web_docs: List[Document]
    ) -> List[Dict[str, str]]:
        """Extract source metadata from documents."""
        sources = []

        # Local sources
        for doc in local_docs:
            sources.append({
                "type": "document",
                "source": Path(doc.metadata.get("source", "Unknown")).name,
                "page": str(doc.metadata.get("page", "?")),
            })

        # Web sources
        for doc in web_docs:
            sources.append({
                "type": "web",
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", ""),
            })

        return sources

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.vector_store.get_collection_stats()
        stats["web_search_enabled"] = self.use_web_search_flag
        stats["top_k"] = self.top_k
        return stats
