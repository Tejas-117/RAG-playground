"""Central logging configuration for backend application events."""

import os
from logging.config import dictConfig

# INFO records normal stage lifecycle events without exposing verbose internals.
DEFAULT_LOG_LEVEL = "INFO"

# Operators can increase or reduce backend verbosity without changing application code.
LOG_LEVEL_ENVIRONMENT_VARIABLE = "BACKEND_LOG_LEVEL"

# Keep local logs readable while retaining the timestamp, severity, and source module.
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Restrict environment input to conventional operational levels supported in Python 3.10.
SUPPORTED_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


def configure_logging() -> None:
    """Configure backend application logs from one environment-controlled level.

    Args:
        None. Configuration reads ``BACKEND_LOG_LEVEL`` from the environment.

    Returns:
        None. The ``backend`` logger hierarchy writes formatted records to stdout.
    """
    configured_level = os.getenv(
        LOG_LEVEL_ENVIRONMENT_VARIABLE,
        DEFAULT_LOG_LEVEL,
    ).upper()

    # Invalid level names fall back to INFO so a typo cannot prevent backend startup.
    resolved_level = (
        configured_level
        if configured_level in SUPPORTED_LOG_LEVELS
        else DEFAULT_LOG_LEVEL
    )

    # Configure only the application logger hierarchy. Existing Uvicorn loggers retain
    # their handlers, which avoids duplicate server access and lifecycle messages.
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "backend": {
                    "format": LOG_FORMAT,
                    "datefmt": LOG_DATE_FORMAT,
                }
            },
            "handlers": {
                "backend_console": {
                    "class": "logging.StreamHandler",
                    "formatter": "backend",
                    "level": resolved_level,
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "backend": {
                    "handlers": ["backend_console"],
                    "level": resolved_level,
                    "propagate": False,
                }
            },
        }
    )
