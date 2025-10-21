from typing import Dict, Any
import json
from core.log.logger import logger
from domain.enums.workflow_state import WorkflowState
from prompts.router_agent_prompt import ROUTER_AGENT_PROMPT


class RouterAgent:
    """Analyzes user query and routes to appropriate path"""

    def __init__(self, llm=None):
        self.llm = llm
        logger.info("RouterAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Analyze query clarity and determine routing using LLM"""
        try:
            user_query = state.get("user_query", "")
            logger.info(f"RouterAgent analyzing query: {user_query}")

            if self.llm is None:
                # Fallback to simple heuristic if no LLM available
                logger.warning("No LLM available, using heuristic fallback")
                if len(user_query.split()) < 3 or "?" not in user_query:
                    query_clarity = "ambiguous"
                    routing_decision = "clarification_agent"
                else:
                    query_clarity = "clear"
                    routing_decision = "intent_analyzer"
                
                return {
                    **state,
                    "original_query": user_query,
                    "query_clarity": query_clarity,
                    "routing_decision": routing_decision,
                    "clarification_needed": query_clarity == "ambiguous"
                }

            # Use LLM for intelligent query analysis
            prompt = ROUTER_AGENT_PROMPT.format(user_query=user_query)
            
            try:
                # Get response from LLM
                response = self.llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parse JSON response
                analysis_result = self._parse_llm_response(response_text)
                
                if analysis_result:
                    query_clarity = analysis_result.get("clarity_assessment", "clear")
                    routing_decision = analysis_result.get("routing_decision", "intent_analyzer")
                    reasoning = analysis_result.get("reasoning", "")
                    confidence_score = analysis_result.get("confidence_score", 0.5)
                    
                    logger.info(f"RouterAgent analysis: {query_clarity} -> {routing_decision} (confidence: {confidence_score})")
                    
                    return {
                        **state,
                        "original_query": user_query,
                        "query_clarity": query_clarity,
                        "routing_decision": routing_decision,
                        "routing_reasoning": reasoning,
                        "routing_confidence": confidence_score,
                        "clarification_needed": query_clarity == "ambiguous"
                    }
                else:
                    # Fallback if JSON parsing fails
                    logger.warning("Failed to parse LLM response, using heuristic fallback")
                    return self._fallback_analysis(state, user_query)
                    
            except Exception as llm_error:
                logger.error(f"LLM analysis failed: {llm_error}")
                return self._fallback_analysis(state, user_query)

        except Exception as e:
            logger.error(f"RouterAgent error: {e}")
            return {**state, "error_message": str(e)}

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract analysis result"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Extract query_analysis if nested
                if "query_analysis" in parsed:
                    return parsed["query_analysis"]
                return parsed
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            
        return None

    def _fallback_analysis(self, state: WorkflowState, user_query: str) -> WorkflowState:
        """Fallback analysis using simple heuristics"""
        # Simple heuristic analysis
        query_lower = user_query.lower().strip()
        
        # Check for ambiguous patterns
        ambiguous_patterns = [
            "help me", "what about", "tell me more", "how do i", 
            "find something", "what's the latest", "can you help"
        ]
        
        is_ambiguous = (
            len(user_query.split()) < 3 or
            any(pattern in query_lower for pattern in ambiguous_patterns) or
            query_lower.endswith("?") and len(user_query.split()) < 4
        )
        
        if is_ambiguous:
            query_clarity = "ambiguous"
            routing_decision = "clarification_agent"
        else:
            query_clarity = "clear"
            routing_decision = "intent_analyzer"
        
        return {
            **state,
            "original_query": user_query,
            "query_clarity": query_clarity,
            "routing_decision": routing_decision,
            "routing_reasoning": "Fallback heuristic analysis",
            "routing_confidence": 0.6,
            "clarification_needed": query_clarity == "ambiguous"
        }


class ClarificationAgent:
    """Asks for clarification when query is ambiguous"""

    def __init__(self, llm=None):
        self.llm = llm
        logger.info("ClarificationAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate clarification questions"""
        try:
            user_query = state.get("user_query", "")
            logger.info(f"ClarificationAgent processing: {user_query}")

            # TODO: Implement actual clarification logic
            refined_query = f"Clarified: {user_query}"

            return {
                **state,
                "refined_query": refined_query,
                "query_clarity": "clear"
            }
        except Exception as e:
            logger.error(f"ClarificationAgent error: {e}")
            return {**state, "error_message": str(e)}


class IntentAnalyzerAgent:
    """Determines intent and decides which retrieval method to use"""

    def __init__(self, llm=None):
        self.llm = llm
        logger.info("IntentAnalyzerAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Analyze intent and make decision"""
        try:
            query = state.get("refined_query") or state.get("user_query", "")
            logger.info(f"IntentAnalyzerAgent analyzing: {query}")

            # TODO: Implement actual intent analysis
            # Simple heuristic for now
            query_lower = query.lower()

            if "document" in query_lower or "pdf" in query_lower:
                decision = "pdf_content"
            elif "latest" in query_lower or "current" in query_lower or "news" in query_lower:
                decision = "external_info"
            else:
                decision = "both"

            return {
                **state,
                "intent": "analyzed",
                "decision": decision
            }
        except Exception as e:
            logger.error(f"IntentAnalyzerAgent error: {e}")
            return {**state, "error_message": str(e)}


class RAGRetrievalAgent:
    """Retrieves information from PDF content using RAG"""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        logger.info("RAGRetrievalAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant information from vector DB"""
        try:
            query = state.get("refined_query") or state.get("user_query", "")
            logger.info(f"RAGRetrievalAgent retrieving for: {query}")

            # TODO: Implement actual RAG retrieval
            rag_results = [
                {"content": "Sample PDF content", "score": 0.95}
            ]

            return {
                **state,
                "rag_results": rag_results
            }
        except Exception as e:
            logger.error(f"RAGRetrievalAgent error: {e}")
            return {**state, "error_message": str(e)}


class WebSearchAgent:
    """Performs web search for external information"""

    def __init__(self, search_api=None):
        self.search_api = search_api
        logger.info("WebSearchAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Perform web search"""
        try:
            query = state.get("refined_query") or state.get("user_query", "")
            logger.info(f"WebSearchAgent searching for: {query}")

            # TODO: Implement actual web search
            web_search_results = [
                {"title": "Sample Result", "snippet": "Sample snippet", "url": "https://example.com"}
            ]

            return {
                **state,
                "web_search_results": web_search_results
            }
        except Exception as e:
            logger.error(f"WebSearchAgent error: {e}")
            return {**state, "error_message": str(e)}


class HybridAgent:
    """Combines RAG and Web Search"""

    def __init__(self, rag_agent=None, web_search_agent=None):
        self.rag_agent = rag_agent or RAGRetrievalAgent()
        self.web_search_agent = web_search_agent or WebSearchAgent()
        logger.info("HybridAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Execute both RAG and Web Search"""
        try:
            logger.info("HybridAgent executing both retrieval methods")

            # Execute RAG retrieval
            state = self.rag_agent(state)

            # Execute Web Search
            state = self.web_search_agent(state)

            return state
        except Exception as e:
            logger.error(f"HybridAgent error: {e}")
            return {**state, "error_message": str(e)}


class VectorDBSearchAgent:
    """Searches vector database for similar content"""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        logger.info("VectorDBSearchAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Search vector database"""
        try:
            query = state.get("refined_query") or state.get("user_query", "")
            logger.info(f"VectorDBSearchAgent searching: {query}")

            # TODO: Implement actual vector DB search
            vector_db_results = [
                {"content": "Vector DB result", "similarity": 0.92}
            ]

            return {
                **state,
                "vector_db_results": vector_db_results
            }
        except Exception as e:
            logger.error(f"VectorDBSearchAgent error: {e}")
            return {**state, "error_message": str(e)}


class ResponseGeneratorAgent:
    """Generates final response based on retrieved information"""

    def __init__(self, llm=None):
        self.llm = llm
        logger.info("ResponseGeneratorAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate response from retrieved information"""
        try:
            logger.info("ResponseGeneratorAgent generating response")

            # Get all retrieved information
            rag_results = state.get("rag_results", [])
            web_results = state.get("web_search_results", [])
            vector_results = state.get("vector_db_results", [])

            # TODO: Implement actual response generation with LLM
            response = f"Generated response based on retrieved information"

            return {
                **state,
                "response": response,
                "final_response": response
            }
        except Exception as e:
            logger.error(f"ResponseGeneratorAgent error: {e}")
            return {**state, "error_message": str(e)}


class SessionMemoryAgent:
    """Updates session memory with conversation history"""

    def __init__(self, memory_store=None):
        self.memory_store = memory_store
        logger.info("SessionMemoryAgent initialized")

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Update session memory"""
        try:
            logger.info("SessionMemoryAgent updating session")

            query = state.get("user_query", "")
            response = state.get("final_response", "")

            # TODO: Implement actual session memory storage
            conversation_history = state.get("conversation_history", [])
            conversation_history.append({
                "query": query,
                "response": response
            })

            return {
                **state,
                "conversation_history": conversation_history
            }
        except Exception as e:
            logger.error(f"SessionMemoryAgent error: {e}")
            return {**state, "error_message": str(e)}