# app/logging_setup.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "teeth_srv") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already set up

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file = os.environ.get("LOG_FILE", "/app/runtime/server.log")
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        logger.warning("Could not set up file logging", exc_info=True)

    logger.propagate = False
    logger.debug("Logger initialised (level=%s, file=%s)", level_name, log_file)
    return logger