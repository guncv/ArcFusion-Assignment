import logging
import os
from pythonjsonlogger import jsonlogger
import coloredlogs

def get_logger(name: str = "arcfusion_logger") -> logging.Logger:
    env = os.getenv("ENV", "dev")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))

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
