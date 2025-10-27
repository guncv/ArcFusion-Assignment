import sys

from enum import Enum
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from src.constants import ArcFusionErrorCodes

class ArcFusionExceptionBase(Exception):
    def __init__(self, error_code: Enum , description: str, *args, **kwargs):
        self.error_code = error_code
        self.pec_code = error_code.value['PEC_CODE']
        self.pec_message = error_code.name
        self.description = description
        self.detail = {'error_code': self.pec_code, 'error_message': self.pec_message, 'description': self.description}
        self.traceback = sys.exc_info()

        msg = '[PC-{0}] {1}: {2}'.format(error_code.value['PEC_CODE'], error_code.name, description)
        super().__init__(msg)
    
    def get_detail(self) -> dict:
        return self.detail

class ArcFusionException(ArcFusionExceptionBase):
    def __init__(self, error_code: ArcFusionErrorCodes, description: str, *args, **kwargs):
        self.error_code_http = error_code.value['HTTP_CODE']
        super().__init__(error_code, description)

    def raise_HTTPException(self):
        raise HTTPException(status_code=self.error_code_http, detail=self.detail)
    
    def convert_to_JSONResponse(self, resp_uuid: str = None) -> JSONResponse:
        content = {"detail": self.detail}
        if resp_uuid:
            content["resp_uuid"] = resp_uuid
        return JSONResponse(status_code=self.error_code_http, content=content)

def convert_to_ArcFusionException(error_code_http: int, detail: dict) -> ArcFusionException:
    enum_name = detail["error_message"]
    enum_value = {'PEC_CODE': detail["pec_code"], 'HTTP_CODE': error_code_http}
    _ArcFusionErrorCodes = Enum("_ArcFusionErrorCodes", {enum_name: enum_value})
    
    error_code = getattr(_ArcFusionErrorCodes, enum_name)
    description = detail["description"]

    return ArcFusionException(error_code=error_code, description=description)