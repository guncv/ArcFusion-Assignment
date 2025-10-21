from pydantic import BaseModel


class HealthCheckResp(BaseModel):
    status: str

class CheckWorkflowReq(BaseModel):
    user_input: str

class CheckWorkflowResp(BaseModel):
    message: str
