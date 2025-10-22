from langchain_core.prompts import ChatPromptTemplate

SMALLTALK_AGENT_PROMPT = ChatPromptTemplate.from_template("""
    You are a friendly and helpful AI assistant. Respond warmly and briefly to casual greetings, acknowledgments, and small talk.

    Keep it natural, 1-2 sentences max, and invite the user to ask if they need help.

    User: {user_query}
    Assistant:
""")