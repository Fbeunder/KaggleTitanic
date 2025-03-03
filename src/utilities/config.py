# -*- coding: utf-8 -*-
"""
Configuration Module

This module provides configuration settings for the Titanic survival prediction project.
"""

import os
import json


# Default configuration
_DEFAULT_CONFIG = {
    'data_dir': 'data',
    'models_dir': 'models',
    'submissions_dir': 'submissions',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'default_model': 'random_forest',
    'web_port': 8050,
    'web_host': '0.0.0.0',
    'debug_mode': True
}

# Global configuration object
_CONFIG = None


def load_config(config_file='config.json'):
    """
    Load configuration from a JSON file.
    
    Args:
        config_file (str, optional): Path to the configuration file.
        
    Returns:
        dict: The loaded configuration.
    """
    global _CONFIG
    
    # Start with default configuration
    _CONFIG = _DEFAULT_CONFIG.copy()
    
    # Try to load configuration from file
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                _CONFIG.update(file_config)
    except Exception as e:
        print(f"Warning: Could not load configuration from {config_file}: {e}")
    
    return _CONFIG


def get_config(refresh=False):
    """
    Get the current configuration.
    
    Args:
        refresh (bool, optional): Whether to reload the configuration from file.
        
    Returns:
        dict: The current configuration.
    """
    global _CONFIG
    
    if _CONFIG is None or refresh:
        load_config()
    
    return _CONFIG


def save_config(config, config_file='config.json'):
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration to save.
        config_file (str, optional): Path to the configuration file.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")
        return False
