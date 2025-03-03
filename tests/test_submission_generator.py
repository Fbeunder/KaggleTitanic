# -*- coding: utf-8 -*-
"""
Tests for the Submission Generator module

This module contains unit tests for the SubmissionGenerator class
which is responsible for generating Kaggle submission files.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.utilities.submission_generator import SubmissionGenerator
from src.data_processing.data_loader import DataLoader


class MockModel:
    """Mock model for testing submission generator"""
    
    def __init__(self, return_value=None):
        self.return_value = return_value if return_value is not None else np.array([0, 1, 0, 1])
    
    def predict(self, X):
        """Mock predict method"""
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value
    
    def __class__(self):
        return MockModel


class TestSubmissionGenerator(unittest.TestCase):
    """Test cases for SubmissionGenerator class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test submissions
        self.temp_dir = tempfile.mkdtemp()
        self.patcher = patch('src.utilities.submission_generator.get_project_root')
        self.mock_get_project_root = self.patcher.start()
        self.mock_get_project_root.return_value = self.temp_dir
        
        # Create submission generator with mocked directory
        self.submission_generator = SubmissionGenerator()
        
        # Create mock test data
        self.mock_test_data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Pclass': [1, 2, 3, 1],
            'Sex': ['male', 'female', 'male', 'female']
        })
        
        # Mock model
        self.mock_model = MockModel()
    
    def tearDown(self):
        """Clean up after tests"""
        self.patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    @patch('src.utilities.submission_generator.DataLoader')
    def test_generate_submission(self, mock_data_loader_class):
        """Test generate_submission method"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Test
        X_test = pd.DataFrame({'col1': [1, 2, 3, 4]})
        result = self.submission_generator.generate_submission(self.mock_model, X_test, file_name="test_submission.csv")
        
        # Assert
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith('test_submission.csv'))
        
        # Check file content
        submission_df = pd.read_csv(result)
        self.assertEqual(list(submission_df.columns), ['PassengerId', 'Survived'])
        self.assertEqual(len(submission_df), 4)
        
        # Check that the file has been properly saved
        self.assertEqual(len(os.listdir(self.submission_generator.full_submissions_dir)), 1)
    
    @patch('src.utilities.submission_generator.DataLoader')
    def test_validate_submission_valid(self, mock_data_loader_class):
        """Test validate_submission with valid submission"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create a valid submission
        submission_df = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Survived': [0, 1, 0, 1]
        })
        
        # Test
        result = self.submission_generator.validate_submission(submission_df=submission_df)
        
        # Assert
        self.assertTrue(result['valid'])
        self.assertEqual(len(result.get('errors', [])), 0)
    
    @patch('src.utilities.submission_generator.DataLoader')
    def test_validate_submission_invalid(self, mock_data_loader_class):
        """Test validate_submission with invalid submission"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create an invalid submission
        submission_df = pd.DataFrame({
            'PassengerId': [1, 2, 3],  # Missing passenger ID
            'Survived': [0, 1, 0]
        })
        
        # Test
        result = self.submission_generator.validate_submission(submission_df=submission_df)
        
        # Assert
        self.assertFalse(result['valid'])
        self.assertGreater(len(result.get('errors', [])), 0)
    
    @patch('src.utilities.submission_generator.DataLoader')
    def test_validate_submission_invalid_values(self, mock_data_loader_class):
        """Test validate_submission with invalid Survived values"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create a submission with invalid values
        submission_df = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Survived': [0, 2, 0, 1]  # 2 is invalid
        })
        
        # Test
        result = self.submission_generator.validate_submission(submission_df=submission_df)
        
        # Assert
        self.assertFalse(result['valid'])
        self.assertTrue(any('Invalid Survived values' in error for error in result.get('errors', [])))
    
    @patch('src.utilities.submission_generator.DataLoader')
    def test_compare_submissions(self, mock_data_loader_class):
        """Test compare_submissions method"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create two submissions
        sub1 = os.path.join(self.submission_generator.full_submissions_dir, "sub1.csv")
        sub2 = os.path.join(self.submission_generator.full_submissions_dir, "sub2.csv")
        
        pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Survived': [0, 1, 0, 1]
        }).to_csv(sub1, index=False)
        
        pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Survived': [0, 1, 1, 0]  # Different predictions
        }).to_csv(sub2, index=False)
        
        # Test
        result = self.submission_generator.compare_submissions([sub1, sub2], names=['Model1', 'Model2'])
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertTrue('Agreement (%)' in result.columns)
        
        # Check agreement rate
        agreement_row = result[result['Comparison'] == 'Model1 vs Model2']
        self.assertEqual(agreement_row['Agreement (%)'].values[0], 50.0)  # 2 of 4 agree
    
    @patch('src.utilities.submission_generator.plt')
    @patch('src.utilities.submission_generator.DataLoader')
    def test_plot_prediction_distribution(self, mock_data_loader_class, mock_plt):
        """Test plot_prediction_distribution method"""
        # Setup mocks
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Pclass': [1, 2, 3, 1],
            'Sex': ['male', 'female', 'male', 'female']
        })
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create a submission file
        submission_path = os.path.join(self.submission_generator.full_submissions_dir, "sub.csv")
        pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Survived': [0, 1, 0, 1]
        }).to_csv(submission_path, index=False)
        
        # Save to instance for testing
        self.submission_generator.last_submission = {
            'path': submission_path
        }
        
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Test
        result = self.submission_generator.plot_prediction_distribution()
        
        # Assert
        self.assertIsNotNone(result)
        mock_plt.subplots.assert_called_once()
        
    @patch('src.utilities.submission_generator.DataLoader')
    def test_list_submissions(self, mock_data_loader_class):
        """Test list_submissions method"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create some submission files
        for i in range(3):
            file_path = os.path.join(self.submission_generator.full_submissions_dir, f"sub{i}.csv")
            pd.DataFrame({
                'PassengerId': [1, 2, 3, 4],
                'Survived': [0, 1, 0, 1]
            }).to_csv(file_path, index=False)
        
        # Test
        result = self.submission_generator.list_submissions()
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue('file_name' in result.columns)
        
    @patch('src.utilities.submission_generator.DataLoader')
    def test_export_submission(self, mock_data_loader_class):
        """Test export_submission method"""
        # Setup mock
        mock_data_loader = MagicMock()
        mock_data_loader.load_test_data.return_value = self.mock_test_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Create a submission
        submission_path = os.path.join(self.submission_generator.full_submissions_dir, "sub.csv")
        pd.DataFrame({
            'PassengerId': [1, 2, 3, 4],
            'Survived': [0, 1, 0, 1]
        }).to_csv(submission_path, index=False)
        
        # Save to instance for testing
        self.submission_generator.last_submission = {
            'path': submission_path
        }
        
        # Create export directory
        export_dir = os.path.join(self.temp_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, "exported.csv")
        
        # Test
        result = self.submission_generator.export_submission(output_path=export_path)
        
        # Assert
        self.assertEqual(result, export_path)
        self.assertTrue(os.path.exists(export_path))
        
        # Check content
        exported_df = pd.read_csv(export_path)
        self.assertEqual(len(exported_df), 4)
        self.assertEqual(list(exported_df.columns), ['PassengerId', 'Survived'])


if __name__ == '__main__':
    unittest.main()
