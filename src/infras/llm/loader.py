from functools import lru_cache
from src.config import config
from src.infras.log import logger

class LlmLoader:
    @lru_cache(maxsize=32)
    def loadLLM(self, type):
        type_key = type.value if hasattr(type, 'value') else type
        llm_config = config[type_key]
        provider = llm_config["api_provider"]
        model = llm_config["model"]

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=llm_config["api_key"],
                model=model,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
            )
        elif provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            llm = ChatDeepSeek(
                api_key=llm_config["api_key"],
                model=model,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                anthropic_api_key=llm_config["api_key"],
                model=model,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"],
            )
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError("Unsupported LLM provider")

        logger.debug(f"LLM instance created successfully for {type_key}")
        return llm

llm_loader = LlmLoader()
