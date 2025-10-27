from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from src.constants.llm_type import LLMType
from src.infras import llm_loader
from src.config import config
from src.prompts import WEB_CONSISTENCY_PROMPT
from src.infras import logger

class WebEvaluator:

    def __init__(self):
        self.llm = llm_loader.loadLLM(LLMType.SYNTHESIZER_AGENT)
        self.consistency_chain = WEB_CONSISTENCY_PROMPT | self.llm | StrOutputParser()

        embedding_config = config.get("embedding", {})
        openai_config = embedding_config.get("openai", {})
        embedding_model = openai_config.get("model_name", "text-embedding-3-small")
        embedder_api_key = openai_config.get("api_key")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=embedder_api_key
        )

    async def evaluate(
        self,
        current_response: str,
        web_results: List[Document],
        user_query: str
    ) -> Dict[str, Any]:
        try:
            web_context = self._format_web_context(web_results)

            consistency = await self._check_consistency(current_response, web_context)
            relevance_score = await self._calculate_relevance(user_query, web_results)
            confidence_score = self._calculate_confidence_score(
                consistency=consistency,
                relevance=relevance_score
            )

            return {
                "factual_consistency": consistency,
                "relevance_score": relevance_score,
                "confidence_score": confidence_score,
                "metadata": {
                    "num_web_results": len(web_results),
                    "tool_type": "web_search"
                }
            }

        except Exception as e:
            logger.error(f"Web evaluation error: {e}")
            return {
                "factual_consistency": "error",
                "relevance_score": 0.0,
                "confidence_score": 0.0,
                "metadata": {"error": str(e)}
            }

    async def _check_consistency(self, answer: str, web_context: str) -> str:
        try:
            result = await self.consistency_chain.ainvoke({
                "web_context": web_context,
                "answer": answer
            })

            result = result.strip().lower()
            if "consistent" in result and "inconsistent" not in result:
                return "consistent"
            elif "partial" in result:
                return "partial"
            elif "unsupported" in result or "inconsistent" in result:
                return "unsupported"
            else:
                return "partial"

        except Exception as e:
            logger.error(f"Consistency check error: {e}")
            return "error"

    async def _calculate_relevance(self, query: str, web_results: List[Document]) -> float:
        if not web_results or not query:
            return 0.0

        try:
            query_embedding = await self.embeddings.aembed_query(query)

            web_texts = [doc.page_content for doc in web_results[:5]]
            if not web_texts:
                return 0.0

            web_embeddings = await self.embeddings.aembed_documents(web_texts)

            similarities = []
            for web_emb in web_embeddings:
                similarity = self._cosine_similarity(query_embedding, web_emb)
                similarities.append(similarity)

            return float(np.mean(similarities)) if similarities else 0.0

        except Exception as e:
            logger.error(f"Relevance calculation error: {e}")
            return 0.5

    def _calculate_confidence_score(
        self,
        consistency: str,
        relevance: float
    ) -> float:
        consistency_score = {
            "consistent": 1.0,
            "partial": 0.5,
            "unsupported": 0.0,
            "error": 0.3
        }.get(consistency, 0.5)

        confidence = (0.6 * consistency_score) + (0.4 * relevance)
        return float(confidence)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _format_web_context(self, web_results: List[Document]) -> str:
        if not web_results:
            return "No web search results available."

        context_parts = []
        for i, doc in enumerate(web_results, 1):
            title = doc.metadata.get("title", "Untitled")
            url = doc.metadata.get("url", "")
            content = doc.page_content

            context_parts.append(
                f"[Web Result {i}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
            )

        return "\n".join(context_parts)