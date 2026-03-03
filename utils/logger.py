"""
utils/logger.py — Structured logging for the gold forecasting system.
"""
import logging
import os
from datetime import datetime

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger writing to both console and file."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (daily rotating)
    date_str = datetime.now().strftime("%Y-%m-%d")
    fh = logging.FileHandler(f"logs/{date_str}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
