from typing import Optional, Dict, Any
from langsmith import Client
import logging

# Use basic logger to avoid circular import
logger = logging.getLogger(__name__)

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
            return

        if not self.api_key or len(self.api_key) < 10:
            self.enabled = False
            logger.warning("[LangSmithTracer] Invalid API key. Tracing disabled.")
            return

        try:
            self.client = Client(api_key=self.api_key, api_url=self.endpoint)
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

_langsmith_tracer: Optional[LangSmithTracer] = None

def get_langsmith_tracer(config: Optional[Dict[str, Any]] = None) -> LangSmithTracer:
    global _langsmith_tracer
    if _langsmith_tracer is None:
        if config is None:
            from src.config import config as app_config
            config = app_config
        _langsmith_tracer = LangSmithTracer(config)
    return _langsmith_tracer


class _LangSmithTracerProxy:
    def __getattr__(self, name):
        return getattr(get_langsmith_tracer(), name)

langsmith_tracer = _LangSmithTracerProxy()