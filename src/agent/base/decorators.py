from functools import wraps
from src.utils import ArcFusionException
from src.constants import ArcFusionErrorCodes

def handle_agent_error(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            if not isinstance(e, ArcFusionException):
                raise ArcFusionException(
                    error_code=ArcFusionErrorCodes.INTERNAL_ERROR,
                    description=f"{self.name} error: [{type(e).__name__}]: {str(e)}",
                )
            raise
    return wrapper
