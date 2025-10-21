from enum import Enum

class ArcFusionErrorCodes(Enum):
    INCORRECT_REQUEST_FORMAT = {'PEC_CODE': 10000, 'HTTP_CODE': 400}
    INTERNAL_ERROR = {'PEC_CODE': 50000, 'HTTP_CODE': 500}

