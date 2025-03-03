# -*- coding: utf-8 -*-
"""
Test Imports Module

This module tests that all project modules can be successfully imported.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestImports(unittest.TestCase):
    """
    TestImports class to verify that all modules can be imported.
    
    This test suite ensures that the basic project structure is sound
    and all module dependencies are properly set up.
    """
    
    def test_data_processing_imports(self):
        """
        Test that data processing modules can be imported.
        """
        from src.data_processing import data_loader, data_preprocessor
        self.assertTrue(True)  # If we got here, the imports succeeded
    
    def test_modelling_imports(self):
        """
        Test that modelling modules can be imported.
        """
        from src.modelling import model_factory, model_trainer, model_evaluator
        self.assertTrue(True)  # If we got here, the imports succeeded
    
    def test_feature_engineering_imports(self):
        """
        Test that feature engineering modules can be imported.
        """
        from src.feature_engineering import feature_creator, feature_selector
        self.assertTrue(True)  # If we got here, the imports succeeded
    
    def test_web_interface_imports(self):
        """
        Test that web interface modules can be imported.
        """
        from src.web_interface import app, dashboard, model_interface
        self.assertTrue(True)  # If we got here, the imports succeeded
    
    def test_utilities_imports(self):
        """
        Test that utilities modules can be imported.
        """
        from src.utilities import config, utils, submission_generator
        self.assertTrue(True)  # If we got here, the imports succeeded


if __name__ == '__main__':
    unittest.main()
