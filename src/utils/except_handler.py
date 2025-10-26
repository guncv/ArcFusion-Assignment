from fastapi import Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError

from domain.enums.error_code import ArcFusionErrorCodes
from core.utils.exception import ArcFusionException

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        detail = exc.errors()
        description = f'{detail[0]["type"]}: {str(detail[0]["loc"])}'

        e = ArcFusionException(error_code=ArcFusionErrorCodes.INCORRECT_REQUEST_FORMAT, description=description)
        return e.convert_to_JSONResponse()
    
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        return e.convert_to_JSONResponse()

async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    try:
        e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=exc.errors())
        return e.convert_to_JSONResponse()
    
    except (ArcFusionException, Exception) as e:
        if type(e) != ArcFusionException:
            e = ArcFusionException(error_code=ArcFusionErrorCodes.INTERNAL_ERROR, description=f"[{type(e).__name__}]: {str(e)}")
        return e.convert_to_JSONResponse()