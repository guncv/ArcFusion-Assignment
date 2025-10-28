from typing import List, Union, Any
from langchain_core.documents import Document

def format_web_context(documents: List[Document]) -> str:
    if not documents:
        return "No web search results available."

    context_parts = ["Web Search Results:\n"]

    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Untitled")
        url = doc.metadata.get("url", "No URL")
        content = doc.page_content

        context_parts.append(
            f"[Web Result {i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )

    return "\n".join(context_parts)

def format_rag_documents_context(
    rag_documents: List[Union[dict, Document, Any]]
) -> str:
    if not rag_documents:
        return "No RAG documents retrieved."

    context_parts = ["RAG Retrieved Documents:\n"]

    for i, doc_item in enumerate(rag_documents, 1):
        if isinstance(doc_item, dict) and "document" in doc_item:
            # Handle RetrievedDocument format with score
            doc = doc_item["document"]
            score = doc_item.get("score", 0.0)
            content = doc.page_content

            context_parts.append(
                f"[RAG Document {i}] (Score: {score:.3f})\n"
                f"Content: {content}\n"
            )
        else:
            # Handle direct Document format
            content = (
                doc_item.page_content
                if hasattr(doc_item, "page_content")
                else str(doc_item)
            )
            score = 0.0
            if hasattr(doc_item, "metadata"):
                score = doc_item.metadata.get("score", 0.0)

            context_parts.append(
                f"[RAG Document {i}] (Score: {score:.3f})\n"
                f"Content: {content}\n"
            )

    return "\n".join(context_parts)