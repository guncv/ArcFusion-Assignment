from typing import Optional, Dict, Any
from langsmith import Client
from src.infras.log import logger

class LangSmithTracer:
    def __init__(self, config: Dict[str, Any]):
        ls_cfg = config.get("langsmith", {})
        self.enabled = ls_cfg.get("enabled", False)
        self.api_key = ls_cfg.get("api_key")
        self.project_name = ls_cfg.get("project_name", "default")
        self.endpoint = ls_cfg.get("endpoint", "https://api.smith.langchain.com")
        self.client = None
        self._initialize()

    def _initialize(self):
        if not self.enabled:
            logger.info("[LangSmithTracer] Disabled via config.")
            return

        if not self.api_key or len(self.api_key) < 10:
            self.enabled = False
            logger.warning("[LangSmithTracer] Invalid API key. Tracing disabled.")
            return

        try:
            self.client = Client(api_key=self.api_key, api_url=self.endpoint)
            logger.info(f"[LangSmithTracer] Connected to {self.endpoint} ({self.project_name})")
        except Exception as e:
            self.enabled = False
            self.client = None
            logger.warning(f"[LangSmithTracer] Initialization failed: {e}")

    @property
    def trace_enabled(self) -> bool:
        return self.enabled and self.client is not None

    def get_trace_url(self, run_id: str) -> Optional[str]:
        if not self.trace_enabled:
            return None
        return f"{self.endpoint}/o/default/projects/p/{self.project_name}/r/{run_id}"

    def add_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.trace_enabled:
            return metadata
        return {"langsmith_project": self.project_name, **metadata}

langsmith_tracer = LangSmithTracer()