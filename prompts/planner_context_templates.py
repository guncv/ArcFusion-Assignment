"""
Context templates for the Planner Agent.

These templates are used to provide different contexts to the planner agent
depending on whether it's an initial planning request or a replanning request
after a previous attempt was insufficient.
"""

INITIAL_PLANNING_CONTEXT_TEMPLATE = """
ORCHESTRATION REQUEST (RAG results were insufficient)

User Query: {user_query}
Refined Query: {refined_query}

RAG RESULTS ANALYSIS:
- RAG has already been executed and was deemed INSUFFICIENT
- The system needs additional information beyond what RAG provided
- Your task is to plan WEB SEARCH queries to supplement RAG results

YOUR TASK:
Plan web search queries to find information that RAG could not provide.
Consider:
- What information is RAG missing?
- What current/real-time data is needed?
- What external sources would complement RAG results?

AVAILABLE TOOLS:
- **web_search**: Search the web for missing information
- **hybrid_search**: Combine RAG + Web (RAG will be re-executed)

RECOMMENDATION: Since RAG was already tried, prefer "web_search" to supplement existing RAG results.
"""

REPLANNING_CONTEXT_TEMPLATE = """
REPLANNING REQUEST (Previous orchestration attempt was insufficient)

User Query: {user_query}
Refined Query: {refined_query}

PREVIOUS ORCHESTRATION ATTEMPT:
- Tools used: {previous_tools}
- Confidence score: {previous_confidence}
- Strategy: {strategy}

CONTEXT:
- RAG was ALREADY executed before orchestration began
- The previous web search strategy did not produce a satisfactory answer
- You need to try a DIFFERENT web search strategy or queries

ANALYSIS:
The previous web search attempt did not complement RAG results sufficiently.
Please choose a DIFFERENT search strategy:

1. Try different search queries or keywords
2. Use hybrid_search to re-execute both RAG and Web with different context
3. Consider what specific information is still missing

Choose tools that will provide ADDITIONAL or DIFFERENT information than the previous attempt.
"""
