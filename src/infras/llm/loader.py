from src.config import config

class LlmLoader:
    def __init__(self):
        self.model = None
        self.provider = None
        self.temperature = None
        self.api_key = None

    def loadLLM(self, type):
        type_key = type.value if hasattr(type, 'value') else type
        self.model = config[type_key]["model"]
        self.provider = config[type_key]["api_provider"]
        self.temperature = config[type_key]["temperature"]
        self.api_key = config[type_key]["api_key"]
        self.max_tokens = config[type_key]["max_tokens"]
        
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                anthropic_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError("Unsupported LLM provider")

llm_loader = LlmLoader()

