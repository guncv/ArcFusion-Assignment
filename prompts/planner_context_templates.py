NO_RAG_CONTEXT_TEMPLATE = """
    DIRECT WEB SEARCH REQUEST (Real-time/Web Data Needed)

    User Query: {user_query}

    SITUATION:
    - Intent Analysis determined this question REQUIRES real-time or web-based information
    - RAG was SKIPPED entirely (no need to search documents for current/live data)
    - You need to go DIRECTLY to web search for fresh information

    EXAMPLES OF THESE QUERIES:
    - "Who is the richest person right now?"
    - "What is the current price of Bitcoin?"
    - "Latest news about [topic]"
    - "Search the web for [topic]"

    YOUR TASK:
    1. Analyze what real-time/web information is needed
    2. Generate 1-3 SPECIFIC search queries to find this information
    3. Each query should target different aspects or sources
    4. Focus on current, up-to-date information

    QUERY GENERATION GUIDELINES:
    - For "current" questions: Use keywords like "latest", "2024", "now", "current"
    - For rankings/comparisons: Include timeframe "2024", "current", "latest"
    - For news: Use "breaking news", "latest updates", "recent developments"
    - Each query should target different angles or sources

    AVAILABLE TOOLS:
    - **web_search**: Generate N queries → spawn N parallel workers (REQUIRED for this scenario)
    - **none**: Only if the question truly cannot be answered

    WORKER ARCHITECTURE:
    - You generate N queries (typically 2-3 for comprehensive coverage)
    - System spawns N workers (one per query)
    - All workers search in PARALLEL for speed
    - Results are combined for a complete answer

    REMEMBER: Since we need real-time data, web_search is almost always the right choice!
    Each query will run in a separate worker for maximum efficiency.
"""

INITIAL_PLANNING_CONTEXT_TEMPLATE = """
    ORCHESTRATION REQUEST (RAG results were insufficient)

    User Query: {user_query}

    RAG REFLECTION FEEDBACK:
        {rag_reflection_comment}

    SITUATION:
    - RAG has already been executed and was deemed INSUFFICIENT or NOT RELEVANT
    - The knowledge base doesn't have enough information to answer this query
    - You need to supplement RAG results with web search

    YOUR TASK:
    1. Think about what specific information is missing from RAG results
    2. Generate 1-3 SPECIFIC search queries to find the missing information
    3. Each query should target different aspects or angles of the information needed
    4. Make queries targeted and precise to get the best results

    QUERY GENERATION GUIDELINES:
    - Break down complex questions into focused search queries
    - Use specific keywords and phrases that would help find authoritative sources
    - Consider different angles: definitions, comparisons, recent updates, examples, etc.
    - Each query should have a clear purpose

    AVAILABLE TOOLS:
    - **web_search**: Generate N queries → spawn N parallel workers (RECOMMENDED)
    - **none**: No search needed (only if truly unnecessary)

    WORKER ARCHITECTURE:
    - You generate N queries (e.g., 3 queries)
    - System spawns N workers (one per query)
    - All workers search in PARALLEL
    - Results are combined for comprehensive answer

    REMEMBER: You MUST provide search_queries when using web_search!
    Each query will run in a separate worker for maximum efficiency.
"""

REPLANNING_CONTEXT_TEMPLATE = """
    REPLANNING REQUEST (Previous orchestration attempt was insufficient)

    User Query: {user_query}

    PREVIOUS ORCHESTRATION ATTEMPT:
    - Previous response: {old_response}

    ORCHESTRATION REFLECTION FEEDBACK AND WHY PREVIOUS ATTEMPT FAILED:
        {reflection_comment}
    
    Previous Workers' Queries:
        {old_queries}

    YOUR TASK:
    1. Analyze the reflection feedback to understand what specific information is missing
    2. Generate TARGETED search queries that address the specific gaps mentioned in the feedback
    3. Focus on finding the missing information rather than avoiding previous queries
    4. Use the feedback to guide your query generation strategy

    REPLANNING STRATEGIES:
    - Use the reflection feedback to identify what specific information is needed
    - Generate queries that directly target the gaps mentioned in the feedback
    - Consider more specific or alternative sources for the missing information
    - Add context or qualifiers based on what the feedback says is missing
    - Focus on authoritative sources that might have the missing details

    Generate TARGETED queries → spawn NEW parallel workers

    WORKER ASSIGNMENT STRATEGY:
    1. Your NEW queries will spawn NEW workers
    2. Each worker runs independently in parallel
    3. More workers = more comprehensive coverage (up to 3 max)
    4. Targeted queries = better chance of finding missing information

    CRITICAL: Use the reflection feedback to guide your query generation!
    Generate 1-3 TARGETED search queries that specifically address what the feedback says is missing.
    Think: What specific information does the feedback say we need? What sources would have that information?
"""
