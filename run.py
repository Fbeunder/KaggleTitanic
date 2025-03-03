#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Titanic Survival Predictor - Main Execution Script

This script acts as the application entry point. It handles path configuration to ensure
that the 'src' package is properly recognized and imported, regardless of whether the app
is run from the root directory or from within the src directory.

Usage:
    python run.py

This will start the Flask web application on http://localhost:5000
"""

import os
import sys

# Ensure that the src package can be imported regardless of where the script is run from
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Now we can safely import from the src package
from src.web_interface.app import app

if __name__ == '__main__':
    print("Starting Titanic Survival Predictor Web Application...")
    print("Open your browser and navigate to http://localhost:5000")
    app.run(debug=True)
