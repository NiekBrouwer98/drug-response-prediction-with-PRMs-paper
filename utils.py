"""
Utility functions for backward compatibility with existing scripts.

This module provides functions that maintain compatibility with the existing
codebase while using the new configuration system.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Import the configuration
from config import config, setup_project


def setup_paths():
    """
    Setup project paths for backward compatibility.
    
    This function mimics the path setup used in existing scripts.
    """
    # Setup project directories
    setup_project()
    
    # Return paths in the format expected by existing scripts
    return {
        'path': os.getcwd(),
        'home_dir': str(config.PROJECT_ROOT),
        'data_dir': str(config.DATA_DIR),
        'resources_dir': str(config.RESOURCES_DIR),
        'results_dir': str(config.RESULTS_DIR),
        'figures_dir': str(config.FIGURES_DIR)
    }


def setup_notebook_imports():
    """
    Setup imports for Jupyter notebooks where __file__ is not defined.
    
    This function should be called at the beginning of notebook cells
    to set up the correct import paths.
    """
    import sys
    from pathlib import Path
    
    # Get the project root directory
    project_root = Path.cwd()
    
    # If you're in a subdirectory, navigate up to find the project root
    while not (project_root / "config.py").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"Project root: {project_root}")
    print(f"Python path includes: {str(project_root)}")
    
    return project_root


def get_script_paths(script_dir: str) -> dict:
    """
    Get paths for a specific script directory.
    
    Args:
        script_dir: Name of the script directory (e.g., '01_process_observed_profiles')
        
    Returns:
        Dictionary with path variables
    """
    paths = setup_paths()
    
    # Add script-specific paths
    if script_dir == '01_process_observed_profiles':
        paths['results_dir'] = str(config.RESULTS_01_DIR)
        paths['figures_dir'] = str(config.FIGURES_01_DIR)
    elif script_dir == '02_process_predicted_profiles':
        paths['results_dir'] = str(config.RESULTS_02_DIR)
        paths['figures_dir'] = str(config.FIGURES_02_DIR)
    elif script_dir == '03_evaluate_predicted_profiles':
        paths['results_dir'] = str(config.RESULTS_03_DIR)
        paths['figures_dir'] = str(config.FIGURES_03_DIR)
    elif script_dir == '04_predict_drug_response':
        paths['results_dir'] = str(config.RESULTS_04_DIR)
        paths['figures_dir'] = str(config.FIGURES_04_DIR)
    
    return paths


def setup_logging_for_script(script_name: str, level: str = "INFO"):
    """
    Setup logging for a specific script.
    
    Args:
        script_name: Name of the script (can be __file__ or a script name)
        level: Logging level
    """
    # Setup project logging
    config.setup_logging()
    
    # Extract just the filename if a full path is provided
    if isinstance(script_name, (str, Path)) and os.path.sep in str(script_name):
        script_name = Path(script_name).stem
    
    # Create script-specific logger
    logger = logging.getLogger(script_name)
    logger.setLevel(getattr(logging, level.upper()))
    # Ensure messages propagate to root logger (which has the file handler)
    logger.propagate = True
    
    return logger


def ensure_directories_exist(*dir_paths):
    """
    Ensure that directories exist, creating them if necessary.
    
    Args:
        *dir_paths: Variable number of directory paths
    """
    for dir_path in dir_paths:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_file_path(base_dir: str, filename: str, create_dir: bool = True) -> str:
    """
    Get full file path and ensure directory exists.
    
    Args:
        base_dir: Base directory path
        filename: Filename
        create_dir: Whether to create directory if it doesn't exist
        
    Returns:
        Full file path
    """
    file_path = Path(base_dir) / filename
    if create_dir:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    return str(file_path)


def validate_data_files(required_files: list, data_dir: str) -> list:
    """
    Validate that required data files exist.
    
    Args:
        required_files: List of required filenames
        data_dir: Data directory path
        
    Returns:
        List of missing files
    """
    missing_files = []
    data_path = Path(data_dir)
    
    for filename in required_files:
        file_path = data_path / filename
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    return missing_files


def log_script_start(script_name: str, logger: Optional[logging.Logger] = None):
    """
    Log the start of a script execution.
    
    Args:
        script_name: Name of the script
        logger: Optional logger instance
    """
    if logger is None:
        logger = setup_logging_for_script(script_name)
    
    logger.info(f"Starting {script_name}")
    logger.info(f"Project root: {config.PROJECT_ROOT}")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Results directory: {config.RESULTS_DIR}")


def log_script_end(script_name: str, logger: Optional[logging.Logger] = None):
    """
    Log the end of a script execution.
    
    Args:
        script_name: Name of the script
        logger: Optional logger instance
    """
    if logger is None:
        logger = setup_logging_for_script(script_name)
    
    logger.info(f"Completed {script_name}")


# Backward compatibility functions
def get_home_dir():
    """Get home directory path (backward compatibility)."""
    return str(config.PROJECT_ROOT)


def get_data_dir():
    """Get data directory path (backward compatibility)."""
    return str(config.DATA_DIR)


def get_results_dir():
    """Get results directory path (backward compatibility)."""
    return str(config.RESULTS_DIR)


def get_figures_dir():
    """Get figures directory path (backward compatibility)."""
    return str(config.FIGURES_DIR)


def get_resources_dir():
    """Get resources directory path (backward compatibility)."""
    return str(config.RESOURCES_DIR)


if __name__ == "__main__":
    # Test the utility functions
    print("Testing utility functions...")
    
    # Setup paths
    paths = setup_paths()
    print("Paths setup successfully:")
    for key, value in paths.items():
        print(f"  {key}: {value}")
    
    # Test script-specific paths
    script_paths = get_script_paths('01_process_observed_profiles')
    print(f"\nScript-specific paths: {script_paths}")
    
    # Test logging
    logger = setup_logging_for_script('test_script')
    logger.info("Test log message")
    
    print("\nUtility functions test completed successfully!")
