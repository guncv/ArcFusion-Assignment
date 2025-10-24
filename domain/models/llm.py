from pydantic import BaseModel

class HealthCheckResp(BaseModel):
    status: str

class LLMRequest(BaseModel):
    user_input: str

class LLMResponse(BaseModel):
    response: str
    rag_synthesizer_response: str
    rag_routing_decision: str
    rag_reflection_comment: str

class ClearHistoryResponse(BaseModel):
    message: str
