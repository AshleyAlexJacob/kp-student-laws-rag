"""Logging configuration for KP Laws Agent.

Configures application loggers to output formatted messages to both
the console and a persistent logs/app.log file.
"""

import logging
import os
from typing import Optional


class LoggerFactory:
    """Factory for creating and configuring standardized application loggers."""

    _log_dir: str = "logs"
    _log_file: str = os.path.join(_log_dir, "app.log")
    _initialized: bool = False

    @classmethod
    def _setup_handlers(cls) -> None:
        """Create log directory and set up root logger handlers once."""
        if cls._initialized:
            return

        # Ensure destination logs directory exists
        os.makedirs(cls._log_dir, exist_ok=True)

        # Standard log format including timestamp, level, module name and message
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Console handler for terminal visibility
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # File handler for persistent logging in logs/app.log
        file_handler = logging.FileHandler(cls._log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Attach handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """Get a configured logger instance with console and file handlers.

        Args:
            name: Name of the logger, typically __name__.

        Returns:
            logging.Logger: Configured logger instance.
        """
        cls._setup_handlers()
        return logging.getLogger(name or "kp_agent")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Convenience helper to retrieve a configured logger.

    Args:
        name: Name of the module or component.

    Returns:
        logging.Logger: Configured logger.
    """
    return LoggerFactory.get_logger(name)

