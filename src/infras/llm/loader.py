from functools import lru_cache
from src.config import config

class LlmLoader:
    @lru_cache(maxsize=32)
    def loadLLM(self, type):
        type_key = type.value if hasattr(type, 'value') else type
        llm_config = config[type_key]

        if llm_config["api_provider"] == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                openai_api_key=llm_config["api_key"],
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
            )
        elif llm_config["api_provider"] == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(
                api_key=llm_config["api_key"],
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
            )
        elif llm_config["api_provider"] == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                anthropic_api_key=llm_config["api_key"],
                model=llm_config["model"],
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
            )
        else:
            raise ValueError("Unsupported LLM provider")

llm_loader = LlmLoader()
