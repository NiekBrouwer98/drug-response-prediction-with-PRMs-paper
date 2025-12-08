"""
Configuration file for Drug Response Prediction with Perturbation Response Models.

This module provides centralized configuration management for the entire project,
including paths, parameters, and environment settings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


class Config:
    """Central configuration class for the drug response prediction project."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        # Set project root directory
        self.PROJECT_ROOT = Path(__file__).parent.absolute()
        
        # Load configuration from file if provided
        if config_file:
            self.load_config(config_file)
        else:
            self._set_default_paths()
            self._set_default_parameters()
            self._set_default_logging()
    
    def _set_default_paths(self):
        """Set default directory paths."""
        # Main directories
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.FIGURES_DIR = self.PROJECT_ROOT / "figures"
        self.RESOURCES_DIR = self.PROJECT_ROOT / "resources"
        self.SCRIPTS_DIR = self.PROJECT_ROOT / "scripts"

        # Subdirectories
        self.GEARS_MODEL = "GEARS"
        self.SCFOUNDATION_MODEL = "scfoundation"
        self.CPA_MODEL = "CPA"
        
        # Results subdirectories
        self.RESULTS_01_DIR = self.RESULTS_DIR / "01_process_measured_profiles"
        self.RESULTS_02_DIR = self.RESULTS_DIR / "02_process_predicted_profiles"
        self.RESULTS_03_DIR = self.RESULTS_DIR / "03_evaluate_predicted_profiles"
        self.RESULTS_04_DIR = self.RESULTS_DIR / "04_predict_drug_response"
        
        # Figures subdirectories
        self.FIGURES_01_DIR = self.FIGURES_DIR / "01_process_measured_profiles"
        self.FIGURES_02_DIR = self.FIGURES_DIR / "02_process_predicted_profiles"
        self.FIGURES_03_DIR = self.FIGURES_DIR / "03_evaluate_predicted_profiles"
        self.FIGURES_04_DIR = self.FIGURES_DIR / "04_predict_drug_response"
    
    def _set_default_parameters(self):
        """Set default analysis parameters."""
        # Data processing parameters
        self.MIN_CELLS_PER_CONDITION = 10
        self.MIN_GENES_PER_CELL = 200
        self.MIN_CELLS_PER_GENE = 3
        
        # Feature selection parameters
        self.DEFAULT_N_FEATURES = 1000
        self.HVG_N_TOP_GENES = 2000
        
        # Cross-validation parameters
        self.DEFAULT_N_OUTER_SPLITS = 5
        self.DEFAULT_N_INNER_SPLITS = 5
        self.RANDOM_STATE = 42
        
        # Model parameters
        self.ELASTICNET_ALPHAS = [0.1, 1.0, 10.0]
        self.ELASTICNET_L1_RATIOS = [0.0, 0.1, 0.5, 1.0]
        self.ELASTICNET_MAX_ITER = 1000
        
        # Drug sensitivity parameters
        self.SENSITIVITY_THRESHOLD = 0.02
        self.IC50_THRESHOLD = 1.0  # μM
        
        # Cell lines and datasets
        self.SCIPLEX_CELL_LINES = ["A549", "K562", "MCF7"]
        self.MCFARLAND_TISSUE_TYPES = [
            "BREAST", "CENTRAL_NERVOUS_SYSTEM", "ENDOMETRIUM", "KIDNEY",
            "LARGE_INTESTINE", "LUNG", "OESOPHAGUS", "OVARY", "PANCREAS",
            "SKIN", "STOMACH", "THYROID", "UPPER_AERODIGESTIVE_TRACT", "URINARY_TRACT"
        ]
        
        # File naming patterns
        self.FILE_PATTERNS = {
            "mcfarland_processed": "mcfarland_{tissue}_{cell_line}.h5ad",
            "sciplex_processed": "sciplex_{cell_line}.h5ad",
            "predictions": "{model}_{cell_line}_{condition_type}.csv",
            "average_effect": "average_effect_{cell_line}_{condition_type}.csv",
            "no_effect": "no_effect_{cell_line}_{condition_type}.csv"
        }
    
    def _set_default_logging(self):
        """Set default logging configuration."""
        self.LOG_LEVEL = logging.INFO
        self.LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.LOG_FILE = self.PROJECT_ROOT / "logs" / "analysis.log"
    
    def load_config(self, config_file: Union[str, Path]):
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update paths if specified
        if 'paths' in config_data:
            for key, value in config_data['paths'].items():
                setattr(self, key.upper(), Path(value))
        
        # Update parameters if specified
        if 'parameters' in config_data:
            for key, value in config_data['parameters'].items():
                setattr(self, key.upper(), value)
    
    def create_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR, self.RESULTS_DIR, self.FIGURES_DIR, self.RESOURCES_DIR,
            self.RESULTS_01_DIR, self.RESULTS_02_DIR, self.RESULTS_03_DIR,
            self.RESULTS_04_DIR,
            self.FIGURES_01_DIR, self.FIGURES_02_DIR, self.FIGURES_03_DIR,
            self.FIGURES_04_DIR,
            self.LOG_FILE.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format=self.LOG_FORMAT,
            handlers=[
                logging.FileHandler(self.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    def get_path(self, path_name: str) -> Path:
        """
        Get a path by name.
        
        Args:
            path_name: Name of the path attribute
            
        Returns:
            Path object
            
        Raises:
            AttributeError: If path_name doesn't exist
        """
        if not hasattr(self, path_name.upper()):
            raise AttributeError(f"Path '{path_name}' not found in configuration")
        return getattr(self, path_name.upper())
    
    def validate_paths(self) -> List[str]:
        """
        Validate that all required paths exist.
        
        Returns:
            List of missing paths
        """
        missing_paths = []
        required_paths = [
            "PROJECT_ROOT", "DATA_DIR", "RESULTS_DIR", "FIGURES_DIR", 
            "RESOURCES_DIR", "SCRIPTS_DIR"
        ]
        
        for path_name in required_paths:
            path = getattr(self, path_name)
            if not path.exists():
                missing_paths.append(str(path))
        
        return missing_paths
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, Path):
                    config_dict[attr_name] = str(attr_value)
                else:
                    config_dict[attr_name] = attr_value
        return config_dict
    
    def save_config(self, output_file: Union[str, Path]):
        """
        Save current configuration to YAML file.
        
        Args:
            output_file: Path to save configuration
        """
        config_dict = self.to_dict()
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Global configuration instance
config = Config()

# Convenience functions for backward compatibility
def get_project_root() -> Path:
    """Get project root directory."""
    return config.PROJECT_ROOT

def get_data_dir() -> Path:
    """Get data directory."""
    return config.DATA_DIR

def get_results_dir() -> Path:
    """Get results directory."""
    return config.RESULTS_DIR

def get_figures_dir() -> Path:
    """Get figures directory."""
    return config.FIGURES_DIR

def get_resources_dir() -> Path:
    """Get resources directory."""
    return config.RESOURCES_DIR

def get_scripts_dir() -> Path:
    """Get scripts directory."""
    return config.SCRIPTS_DIR

def setup_project():
    """Setup project directories and logging."""
    config.create_directories()
    config.setup_logging()
    return config


if __name__ == "__main__":
    # Example usage
    cfg = setup_project()
    print("Project configuration loaded successfully!")
    print(f"Project root: {cfg.PROJECT_ROOT}")
    print(f"Data directory: {cfg.DATA_DIR}")
    print(f"Results directory: {cfg.RESULTS_DIR}")
    
    # Validate paths
    missing = cfg.validate_paths()
    if missing:
        print(f"Missing paths: {missing}")
    else:
        print("All required paths exist!")
