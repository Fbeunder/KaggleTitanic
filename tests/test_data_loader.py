# -*- coding: utf-8 -*-
"""
Unit tests for the DataLoader class.

This module contains tests for the functionality of the DataLoader class.
"""

import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data_processing.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """
    Test cases for the DataLoader class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a temporary test directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create sample test data
        self.sample_train_data = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 1],
            'Pclass': [3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 'Heikkinen, Miss. Laina'],
            'Sex': ['male', 'female', 'female'],
            'Age': [22.0, 38.0, 26.0],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
            'Fare': [7.25, 71.2833, 7.925],
            'Cabin': [None, 'C85', None],
            'Embarked': ['S', 'C', 'S']
        })
        
        self.sample_test_data = pd.DataFrame({
            'PassengerId': [892, 893, 894],
            'Pclass': [3, 3, 2],
            'Name': ['Kelly, Mr. James', 'Wilkes, Mrs. James (Ellen Needs)', 'Myles, Mr. Thomas Francis'],
            'Sex': ['male', 'female', 'male'],
            'Age': [34.5, 47.0, 62.0],
            'SibSp': [0, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['330911', '363272', '240276'],
            'Fare': [7.8292, 7.0, 9.6875],
            'Cabin': [None, None, None],
            'Embarked': ['Q', 'S', 'Q']
        })
        
        # Save sample data to files
        self.train_path = os.path.join(self.test_dir, 'train.csv')
        self.test_path = os.path.join(self.test_dir, 'test.csv')
        self.sample_train_data.to_csv(self.train_path, index=False)
        self.sample_test_data.to_csv(self.test_path, index=False)
        
        # Initialize DataLoader with test directory
        self.data_loader = DataLoader(data_dir=self.test_dir)
    
    def tearDown(self):
        """
        Clean up test fixtures.
        """
        # Clean up test files
        if os.path.exists(self.train_path):
            os.remove(self.train_path)
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_init_with_custom_directory(self):
        """
        Test initializing DataLoader with a custom directory.
        """
        data_loader = DataLoader(data_dir=self.test_dir)
        self.assertEqual(data_loader.data_dir, self.test_dir)
    
    @patch('src.utilities.config.get_config')
    def test_init_with_config_directory(self, mock_get_config):
        """
        Test initializing DataLoader with a directory from config.
        """
        mock_get_config.return_value = {'data_dir': self.test_dir}
        data_loader = DataLoader()
        self.assertEqual(data_loader.data_dir, self.test_dir)
    
    def test_load_train_data(self):
        """
        Test loading training data.
        """
        train_data = self.data_loader.load_train_data()
        
        # Check if data is loaded correctly
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertEqual(len(train_data), 3)
        self.assertEqual(train_data.shape[1], 12)
        
        # Check if specific values are correct
        self.assertEqual(train_data.iloc[0]['PassengerId'], 1)
        self.assertEqual(train_data.iloc[0]['Survived'], 0)
        self.assertEqual(train_data.iloc[0]['Sex'], 'male')
    
    def test_load_test_data(self):
        """
        Test loading testing data.
        """
        test_data = self.data_loader.load_test_data()
        
        # Check if data is loaded correctly
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(len(test_data), 3)
        self.assertEqual(test_data.shape[1], 11)  # No 'Survived' column
        
        # Check if specific values are correct
        self.assertEqual(test_data.iloc[0]['PassengerId'], 892)
        self.assertEqual(test_data.iloc[0]['Pclass'], 3)
        self.assertEqual(test_data.iloc[0]['Sex'], 'male')
    
    def test_get_combined_data(self):
        """
        Test combining training and testing data.
        """
        combined_data, train_samples = self.data_loader.get_combined_data()
        
        # Check if data is combined correctly
        self.assertIsInstance(combined_data, pd.DataFrame)
        self.assertEqual(len(combined_data), 6)  # 3 train + 3 test samples
        self.assertEqual(train_samples, 3)
        
        # Check if indicator column is added
        self.assertIn('is_train', combined_data.columns)
        self.assertEqual(combined_data.iloc[0]['is_train'], 1)  # Train data
        self.assertEqual(combined_data.iloc[3]['is_train'], 0)  # Test data
    
    def test_file_not_found(self):
        """
        Test handling of file not found error.
        """
        # Remove test files
        os.remove(self.train_path)
        
        # Attempt to load missing file
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_train_data()
    
    @patch('pandas.read_csv')
    def test_empty_file(self, mock_read_csv):
        """
        Test handling of empty file error.
        """
        mock_read_csv.side_effect = pd.errors.EmptyDataError("Empty CSV file")
        
        with self.assertRaises(pd.errors.EmptyDataError):
            self.data_loader.load_train_data()
    
    @patch('pandas.read_csv')
    def test_parse_error(self, mock_read_csv):
        """
        Test handling of parse error.
        """
        mock_read_csv.side_effect = pd.errors.ParserError("Parse error")
        
        with self.assertRaises(pd.errors.ParserError):
            self.data_loader.load_train_data()


if __name__ == '__main__':
    unittest.main()
