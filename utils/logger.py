"""
utils/logger.py — ASCII-safe logging for production use.

Fixes Windows UnicodeEncodeError caused by arrow symbols, Rs. symbol, and emojis.
All special characters replaced with ASCII-safe equivalents.
"""
import logging
import os
import sys
from datetime import datetime


class AsciiSafeFormatter(logging.Formatter):
    """Replace non-ASCII characters that crash Windows console encoding."""

    REPLACEMENTS = {
        "\u20b9": "Rs.",      # Rupee sign
        "\u2192": "->",       # Arrow
        "\u2714": "[OK]",
        "\u26a0": "[WARN]",
        "\u274c": "[ERR]",
        "\u2705": "[DONE]",
        "\u23f3": "[...]",
    }

    def format(self, record):
        msg = super().format(record)
        for char, replacement in self.REPLACEMENTS.items():
            msg = msg.replace(char, replacement)
        try:
            msg.encode(sys.stdout.encoding or "ascii")
        except (UnicodeEncodeError, LookupError):
            msg = msg.encode("ascii", errors="replace").decode("ascii")
        return msg


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger writing to both console and file (ASCII-safe)."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt_str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    formatter = AsciiSafeFormatter(fmt_str)

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (UTF-8, non-crashing)
    date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        fh = logging.FileHandler(f"logs/{date_str}.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass  # HF Spaces may have read-only FS in some paths

    return logger
