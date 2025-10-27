import logging
import os
from pythonjsonlogger import jsonlogger
import coloredlogs

def get_logger() -> logging.Logger:
    env = os.getenv("ENV", "dev")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger("arcfusion")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()

        if env == "dev":
            formatter = coloredlogs.ColoredFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = get_logger()
