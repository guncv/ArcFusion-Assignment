from pydantic import BaseModel

class HealthCheckResp(BaseModel):
    status: str

class LLMRequest(BaseModel):
    user_input: str

class LLMResponse(BaseModel):
    response: str

class ClearHistoryResponse(BaseModel):
    message: str