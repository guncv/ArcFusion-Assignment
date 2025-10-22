from pydantic import BaseModel, Field
from typing import Optional
import uuid

class HealthCheckResp(BaseModel):
    status: str

class LLMRequest(BaseModel):
    user_input: str

class LLMResponse(BaseModel):
    message: str

class ClearHistoryResponse(BaseModel):
    message: str