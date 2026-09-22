"""
Utility helpers shared by pipeline scripts.

Thin wrappers around ``config`` for logging and path setup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from config import config, setup_project


def setup_logging_for_script(script_name: str, level: str = "INFO"):
    """Setup a script-specific logger that propagates to the project root logger."""
    config.setup_logging()

    if isinstance(script_name, (str, Path)) and os.path.sep in str(script_name):
        script_name = Path(script_name).stem

    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = True
    return logger


def ensure_directories_exist(*dir_paths):
    """Create directories if they do not already exist."""
    for dir_path in dir_paths:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def log_script_start(script_name: str, logger: Optional[logging.Logger] = None):
    """Log the start of a script execution."""
    if logger is None:
        logger = setup_logging_for_script(script_name)

    logger.info(f"Starting {script_name}")
    logger.info(f"Project root: {config.PROJECT_ROOT}")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Results directory: {config.RESULTS_DIR}")


def log_script_end(script_name: str, logger: Optional[logging.Logger] = None):
    """Log the end of a script execution."""
    if logger is None:
        logger = setup_logging_for_script(script_name)

    logger.info(f"Completed {script_name}")
