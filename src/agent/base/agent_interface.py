from abc import ABC, abstractmethod
from src.graph import WorkflowState

class AgentInterface(ABC):
    
    @abstractmethod
    async def ainvoke(self, state: WorkflowState) -> WorkflowState:
        pass

    def __repr__(self) -> str:
        # String representation of the agent.
        return f"{self.__class__.__name__}()"

    @property
    def name(self) -> str:
        # Get the agent's name.
        return self.__class__.__name__

agent_interface = AgentInterface()
