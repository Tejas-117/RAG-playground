"""Tests for environment-controlled backend logging configuration."""

from unittest.mock import patch

from backend.logging_config import configure_logging


def test_configure_logging_uses_requested_supported_level() -> None:
    """Verify a supported environment level configures the backend hierarchy.

    Args:
        None.

    Returns:
        None. Assertions inspect the dictionary passed to standard logging.
    """
    with (
        patch.dict("os.environ", {"BACKEND_LOG_LEVEL": "debug"}),
        patch("backend.logging_config.dictConfig") as configure_dictionary,
    ):
        configure_logging()

    configuration = configure_dictionary.call_args.args[0]
    assert configuration["loggers"]["backend"]["level"] == "DEBUG"
    assert configuration["handlers"]["backend_console"]["level"] == "DEBUG"


def test_configure_logging_falls_back_for_invalid_level() -> None:
    """Verify an invalid environment value cannot prevent safe INFO logging.

    Args:
        None.

    Returns:
        None. Assertions confirm the documented fallback level.
    """
    with (
        patch.dict("os.environ", {"BACKEND_LOG_LEVEL": "verbose"}),
        patch("backend.logging_config.dictConfig") as configure_dictionary,
    ):
        configure_logging()

    configuration = configure_dictionary.call_args.args[0]
    assert configuration["loggers"]["backend"]["level"] == "INFO"
    assert configuration["handlers"]["backend_console"]["level"] == "INFO"
