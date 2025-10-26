from src.config import config

class LlmLoader:
    def __init__(self):
        self.model = None
        self.provider = None
        self.temperature = None
        self.api_key = None

    def loadLLM(self, type: str):
        self.model = config[type]["model"]
        self.provider = config[type]["api_provider"]
        self.temperature = config[type]["temperature"]
        self.api_key = config[type]["api_key"]
        self.max_tokens = config[type]["max_tokens"]
        
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

