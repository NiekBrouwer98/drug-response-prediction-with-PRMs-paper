"""
Robust import setup for Jupyter notebooks.

This file provides flexible import setup for running the drug response prediction
code in Jupyter notebooks where __file__ is not defined.

Usage:
    # Basic setup
    %run notebook_imports.py
    
    # Or import specific functions
    from notebook_imports import import_functions
    import_functions(['get_McFarland_mean_data', 'get_McFarland_sensitivityinfo'])
    
    # Or import from specific modules
    from notebook_imports import import_from_module
    import_from_module('scripts.sensitivity_predictions', ['get_McFarland_mean_data'])
"""

import sys
from pathlib import Path
import os
from typing import List, Dict, Any, Optional

# Global variable to store the project root
PROJECT_ROOT = None

sys.path.append(str(Path(__file__).parent.parent))
from config import config

def get_project_config():
    return config

def setup_project_root():
    """Setup the project root directory and add it to Python path."""
    global PROJECT_ROOT
    
    # Get the project root directory
    project_root = Path.cwd()
    
    # If you're in a subdirectory, navigate up to find the project root
    # Look for the presence of config.py to identify the project root
    while not (project_root / "config.py").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    PROJECT_ROOT = project_root
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path includes: {str(project_root)}")
    
    return project_root

def import_functions(function_names: List[str], module_path: str = None) -> Dict[str, Any]:
    """
    Import specific functions from modules.
    
    Args:
        function_names: List of function names to import
        module_path: Optional module path (e.g., 'scripts.sensitivity_predictions')
                    If None, will try to find the functions in common modules
    
    Returns:
        Dictionary mapping function names to imported functions
    """
    if PROJECT_ROOT is None:
        setup_project_root()
    
    imported_functions = {}
    
    # Common module paths to search
    common_modules = [
        'scripts.prediction_utils',
        'scripts.process_utils',
        'scripts.preprocess_utils',
        'scripts.mcfarland_data_loaders',
        'scripts.sciplex_data_loaders',
        'config',
        'utils'
    ]
    
    # If specific module path provided, use it
    if module_path:
        modules_to_search = [module_path]
    else:
        modules_to_search = common_modules
    
    for function_name in function_names:
        function_imported = False
        
        for module_path in modules_to_search:
            try:
                # Import the module
                module = __import__(module_path, fromlist=[function_name])
                
                # Check if the function exists in the module
                if hasattr(module, function_name):
                    imported_functions[function_name] = getattr(module, function_name)
                    print(f"✅ Imported {function_name} from {module_path}")
                    function_imported = True
                    break
                    
            except (ImportError, AttributeError):
                continue
        
        if not function_imported:
            print(f"❌ Could not find function {function_name} in any of the searched modules")
    
    return imported_functions

def import_from_module(module_path: str, function_names: List[str]) -> Dict[str, Any]:
    """
    Import specific functions from a specific module.
    
    Args:
        module_path: Module path (e.g., 'scripts.sensitivity_predictions')
        function_names: List of function names to import
    
    Returns:
        Dictionary mapping function names to imported functions
    """
    if PROJECT_ROOT is None:
        setup_project_root()
    
    imported_functions = {}
    
    try:
        # Import the module
        module = __import__(module_path, fromlist=function_names)
        
        for function_name in function_names:
            if hasattr(module, function_name):
                imported_functions[function_name] = getattr(module, function_name)
                print(f"✅ Imported {function_name} from {module_path}")
            else:
                print(f"❌ Function {function_name} not found in {module_path}")
                
    except ImportError as e:
        print(f"❌ Could not import module {module_path}: {e}")
    
    return imported_functions

def import_common_functions():
    """Import commonly used functions from the project."""
    if PROJECT_ROOT is None:
        setup_project_root()
    
    common_functions = [
        'get_McFarland_mean_data',
        'get_McFarland_sensitivityinfo', 
        'get_sens_labels',
        'filter_on_coefficient_of_variation',
        'add_y_and_normalize',
        'get_sciplex_mean_data',
        'get_sciplex_AUCs',
        'get_tissue_labels'
    ]
    
    return import_functions(common_functions)

def import_config_and_utils():
    """Import configuration and utility functions."""
    if PROJECT_ROOT is None:
        setup_project_root()
    
    config_functions = ['config', 'setup_project']
    utils_functions = ['setup_logging_for_script', 'log_script_start', 'log_script_end', 'ensure_directories_exist']
    
    imported = {}
    imported.update(import_from_module('config', config_functions))
    imported.update(import_from_module('utils', utils_functions))
    
    return imported

def list_available_functions(module_path: str) -> List[str]:
    """
    List all available functions in a module.
    
    Args:
        module_path: Module path (e.g., 'scripts.sensitivity_predictions')
    
    Returns:
        List of function names available in the module
    """
    if PROJECT_ROOT is None:
        setup_project_root()
    
    try:
        module = __import__(module_path, fromlist=['*'])
        functions = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
        print(f"📋 Available functions in {module_path}:")
        for func in sorted(functions):
            print(f"  - {func}")
        return functions
    except ImportError as e:
        print(f"❌ Could not import module {module_path}: {e}")
        return []

def quick_setup():
    """Quick setup with common imports."""
    if PROJECT_ROOT is None:
        setup_project_root()
    
    print("🚀 Setting up common imports...")
    
    # Import config and utils
    config_utils = import_config_and_utils()
    
    # Import common data functions
    data_functions = import_common_functions()
    
    # Combine all imports
    all_imports = {**config_utils, **data_functions}
    
    print(f"✅ Setup complete! Imported {len(all_imports)} functions")
    return all_imports

# Auto-setup when the file is run
if __name__ == "__main__":
    setup_project_root()
    print("🎯 Use quick_setup() to import common functions")
    print("🎯 Use import_functions(['func1', 'func2']) to import specific functions")
    print("🎯 Use import_from_module('module.path', ['func1']) to import from specific module")


