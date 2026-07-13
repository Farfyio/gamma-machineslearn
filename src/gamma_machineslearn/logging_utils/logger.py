"""Central logging setup helpers."""

import logging
from pathlib import Path



def configure_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure and return the project logger."""

    logger = logging.getLogger("gamma_machineslearn")
    logger.setLevel(level.upper())

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
