from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document as LlamaDocument

class BaseReader(ABC):

    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> List[LlamaDocument]:
        pass

    @property
    @abstractmethod
    def reader_name(self) -> str:
        pass
