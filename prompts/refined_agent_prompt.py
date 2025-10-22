from langchain_core.prompts import ChatPromptTemplate

REFINED_QUERY_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that refines user queries to make them more specific and clear.

User query: "{user_query}"

Refine the user query to make it more specific and clear.
""")