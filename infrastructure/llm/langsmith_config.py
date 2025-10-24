from typing import Optional, Dict, Any
from langsmith import Client
from core.config.config import nested_config as config
from core.log.logger import logger

class LangSmithTracer:

    def __init__(self):
        self.enabled = config.get("langsmith", {}).get("enabled", False)
        self.api_key = config.get("langsmith", {}).get("api_key", "")
        self.project_name = config.get("langsmith", {}).get("project_name", "arcfusion-dev")
        self.endpoint = config.get("langsmith", {}).get("endpoint", "https://api.smith.langchain.com")
        self.client = None

        if self.enabled:
            if not self.api_key or len(self.api_key) < 10:
                self.enabled = False
            else:
                try:
                    self.client = Client(api_key=self.api_key, api_url=self.endpoint)
                except Exception as e:
                    logger.error(f"[LangSmith] Failed to initialize client: {e}")
                    self.enabled = False
                    self.client = None
        else:
            self.client = None

    def get_trace_url(self, run_id: str) -> Optional[str]:
        if not self.enabled or not self.client:
            return None
        return f"{self.endpoint}/o/default/projects/p/{self.project_name}/r/{run_id}"

    def add_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or not self.client:
            return metadata
        return {
            "langsmith_project": self.project_name,
            **metadata
        }
