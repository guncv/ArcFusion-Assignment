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