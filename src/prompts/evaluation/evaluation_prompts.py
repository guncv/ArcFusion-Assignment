from langchain_core.prompts import ChatPromptTemplate

# RAG Faithfulness/Groundedness Evaluation
RAG_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge that checks if an answer is grounded in the provided context.

    ## Your Task
    Determine if the answer is fully supported by the retrieved documents.

    ## Evaluation Criteria
    - **supported**: All claims in the answer are directly supported by the context
    - **partial**: Some claims are supported, but others are not or are inferred
    - **unsupported**: The answer contains claims not found in the context or contradicts it

    ## Instructions
    1. Carefully read the retrieved documents
    2. Analyze each claim in the answer
    3. Check if each claim is explicitly stated in the documents
    4. Respond with ONLY one word: "supported", "partial", or "unsupported"

    ## Context

    ### Retrieved Documents:
    {rag_context}

    ### Answer to Evaluate:
    {answer}

    Respond with ONLY: supported, partial, or unsupported"""),
    ("human", "Evaluate the faithfulness of this answer.")
])

# WebSearch Factual Consistency Evaluation
WEB_CONSISTENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge that checks if an answer is factually consistent with web search results.

    ## Your Task
    Determine if the answer is factually consistent with the web search snippets.

    ## Evaluation Criteria
    - **consistent**: All facts in the answer align with the web search results
    - **partial**: Some facts align, but others are missing or uncertain
    - **unsupported**: The answer contains facts that contradict or are not found in web results

    ## Instructions
    1. Carefully read the web search results
    2. Analyze each factual claim in the answer
    3. Check if each fact is consistent with the web snippets
    4. Respond with ONLY one word: "consistent", "partial", or "unsupported"

    ## Context

    ### Web Search Results:
    {web_context}

    ### Answer to Evaluate:
    {answer}

    Respond with ONLY: consistent, partial, or unsupported"""),
    ("human", "Evaluate the factual consistency of this answer.")
])

# Answer Relevance Evaluation (for both RAG and Web)
ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge that checks if an answer properly addresses the user's query.

    ## Your Task
    Determine if the answer is relevant and directly addresses what the user asked.

    ## Evaluation Criteria
    - **relevant**: The answer directly and comprehensively addresses the query
    - **partial**: The answer is somewhat related but doesn't fully address the query
    - **irrelevant**: The answer does not address the query or is off-topic

    ## Instructions
    1. Carefully read the user's query
    2. Analyze if the answer addresses the specific question asked
    3. Check if the answer is complete and on-topic
    4. Respond with ONLY one word: "relevant", "partial", or "irrelevant"

    ## Context

    ### User Query:
    {query}

    ### Answer to Evaluate:
    {answer}

    Respond with ONLY: relevant, partial, or irrelevant"""),
    ("human", "Evaluate the relevance of this answer to the query.")
])

# Context Precision Evaluation (RAG only)
CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluation judge that checks if retrieved documents are precisely relevant to the query.

    ## Your Task
    Determine if the retrieved documents are focused and directly relevant to answering the query.

    ## Evaluation Criteria
    - **precise**: All retrieved documents are highly relevant and focused on the query
    - **moderate**: Most documents are relevant, but some contain off-topic information
    - **unfocused**: Many documents are tangentially related or contain irrelevant information

    ## Instructions
    1. Carefully read the user's query
    2. Analyze each retrieved document
    3. Check if documents contain information directly useful for answering the query
    4. Respond with ONLY one word: "precise", "moderate", or "unfocused"

    ## Context

    ### User Query:
    {query}

    ### Retrieved Documents:
    {rag_context}

    Respond with ONLY: precise, moderate, or unfocused"""),
    ("human", "Evaluate the precision of the retrieved context.")
])