INITIAL_PLANNING_TEMPLATE = """
   PLANNING REQUEST (Autonomous Tool Selection)

   User Query: {user_query}

    SITUATION:
    - You are an autonomous planner that decides which tools to use
    - You have access to BOTH internal documents (RAG) and external web search
    - Analyze the query and decide the BEST strategy

    YOUR AVAILABLE TOOLS:

    1. **rag_search** - Search internal knowledge base/documents
       - Use for: Policies, procedures, internal documentation, company info
       - Examples: "What's our refund policy?", "How does our system work?", "Company guidelines"
       - Strengths: Accurate, authoritative for internal information
       - Limitations: No real-time data, no external/current information

    2. **web_search** - Search the web for current information
       - Use for: Real-time data, current events, external information, latest news
       - Examples: "Who is the richest person now?", "Latest AI news", "Current Bitcoin price"
       - Strengths: Up-to-date, broad coverage, external sources
       - Limitations: No access to internal documents

    3. **none** - No search needed
       - Use only if the question truly cannot be answered with available tools

    DECISION-MAKING GUIDELINES:

    **When to use rag_search:**
    - Questions about internal policies, procedures, documentation
    - Company-specific information
    - Technical documentation
    - Established knowledge that doesn't change frequently

    **When to use web_search:**
    - "Current", "latest", "now", "today" in query
    - Real-time data (prices, rankings, news)
    - External information not in internal docs
    - Recent events or developments

    **Strategy Selection:**
    - Choose ONE tool that best fits the query
    - Generate 1-3 diverse queries for that tool
    - Each query should target different aspects (NOT repetitive variations)

    YOUR TASK:
    1. Analyze the query type and information needed
    2. Decide which tool is BEST suited
    3. Generate 1-3 DIVERSE search queries
    4. Each query should approach the problem from a DIFFERENT angle

    REMEMBER: You are AUTONOMOUS - decide based on the query, not hard-coded rules!
"""

# Removed - Planner now autonomously decides tools from the start

REPLANNING_CONTEXT_TEMPLATE = """
   REPLANNING REQUEST (Previous attempt was insufficient - Autonomous Replanning)

   User Query: {user_query}

   PREVIOUS ATTEMPT:
   - Tool used: {previous_tool}
   - Previous response: {old_response}
   - Previous queries: {old_queries}

   REFLECTION FEEDBACK (What's missing):
      {reflection_comment}

   YOUR AVAILABLE TOOLS (Choose the BEST one for replanning):

      1. **rag_search** - Search internal knowledge base/documents
      2. **web_search** - Search the web for current information
      3. **none** - Give up (only if truly impossible)

   REPLANNING STRATEGIES:

   **Analyze the feedback:**
      - What specific information is missing?
      - Did the previous tool fail to find needed information?
      - Would a DIFFERENT tool be better?

   **Tool Switching:**
      - If previous tool was **rag_search** and feedback mentions "missing recent updates" or "needs current data" → Consider **web_search**
      - If previous tool was **web_search** and feedback mentions "needs specific documentation" → Consider **rag_search** (if not tried yet)
      - If same tool but different queries needed → Use same tool with BETTER queries

    **Query Generation:**
      - Generate 1-3 TARGETED queries that address the specific gaps in the feedback
      - Make queries MORE SPECIFIC based on what's missing
      - Each query should target a different aspect of the missing information

   YOUR TASK:
      1. Analyze reflection feedback to identify gaps
      2. Decide if same tool with better queries OR switch to different tool
      3. Generate TARGETED search queries addressing the gaps

   REMEMBER: You're AUTONOMOUS - you can switch tools if the feedback suggests it!
   Generate 1-3 TARGETED search queries that specifically address what the feedback says is missing.
   Think: What specific information does the feedback say we need? What sources would have that information?
"""
