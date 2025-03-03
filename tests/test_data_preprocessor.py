# -*- coding: utf-8 -*-
"""
Unit tests for the DataPreprocessor class.

This module contains tests for the functionality of the DataPreprocessor class.
"""

import unittest
import pandas as pd
import numpy as np
from src.data_processing.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """
    Test cases for the DataPreprocessor class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 0, 1],
            'Pclass': [3, 1, 3, 2, 3],
            'Name': [
                'Braund, Mr. Owen Harris', 
                'Cumings, Mrs. John Bradley (Florence Briggs Thayer)', 
                'Heikkinen, Miss. Laina', 
                'Futrelle, Mrs. Jacques Heath (Lily May Peel)', 
                'Allen, Mr. William Henry'
            ],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, np.nan],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
            'Cabin': [None, 'C85', None, 'C123', None],
            'Embarked': ['S', 'C', 'S', 'S', np.nan],
            'is_train': [1, 1, 1, 1, 1]
        })
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
    
    def test_handle_missing_values(self):
        """
        Test handling of missing values.
        """
        # Count missing values before
        missing_before = self.sample_data.isnull().sum()
        
        # Process data
        processed_data = self.preprocessor.handle_missing_values(self.sample_data)
        
        # Count missing values after
        missing_after = processed_data.isnull().sum()
        
        # Check if missing values are handled
        self.assertEqual(missing_after['Age'], 0, "Missing Age values should be imputed")
        self.assertEqual(missing_after['Embarked'], 0, "Missing Embarked values should be imputed")
        
        # Check if other columns remain unchanged
        self.assertEqual(missing_after['Cabin'], missing_before['Cabin'], "Cabin missing values should remain unchanged")
    
    def test_encode_categorical(self):
        """
        Test encoding of categorical features.
        """
        # Handle missing values first to avoid encoding errors
        data_no_missing = self.preprocessor.handle_missing_values(self.sample_data)
        
        # Encode categorical features
        encoded_data = self.preprocessor.encode_categorical(data_no_missing)
        
        # Check Sex encoding
        self.assertIn('Sex', encoded_data.columns, "Sex column should be present")
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded_data['Sex']), "Sex should be numerically encoded")
        
        # Check Embarked one-hot encoding
        self.assertNotIn('Embarked', encoded_data.columns, "Embarked column should be removed")
        self.assertIn('Embarked_C', encoded_data.columns, "Embarked_C column should be present")
        self.assertIn('Embarked_S', encoded_data.columns, "Embarked_S column should be present")
        
        # Check Pclass one-hot encoding
        self.assertNotIn('Pclass', encoded_data.columns, "Pclass column should be removed")
        self.assertIn('Pclass_1', encoded_data.columns, "Pclass_1 column should be present")
        self.assertIn('Pclass_2', encoded_data.columns, "Pclass_2 column should be present")
        self.assertIn('Pclass_3', encoded_data.columns, "Pclass_3 column should be present")
    
    def test_scale_numeric(self):
        """
        Test scaling of numeric features.
        """
        # Process data
        scaled_data = self.preprocessor.scale_numeric(self.sample_data)
        
        # Check Age scaling
        self.assertTrue('Age' in scaled_data.columns, "Age column should be present")
        if not self.sample_data['Age'].isnull().all():  # Only test if there are non-null values
            self.assertAlmostEqual(scaled_data['Age'].mean(), 0, delta=1e-10, 
                                 msg="Scaled Age should have mean close to 0")
            self.assertAlmostEqual(scaled_data['Age'].std(), 1, delta=1e-10, 
                                 msg="Scaled Age should have std close to 1")
        
        # Check Fare scaling
        self.assertTrue('Fare' in scaled_data.columns, "Fare column should be present")
        self.assertAlmostEqual(scaled_data['Fare'].mean(), 0, delta=1e-10, 
                             msg="Scaled Fare should have mean close to 0")
        self.assertAlmostEqual(scaled_data['Fare'].std(), 1, delta=1e-10, 
                             msg="Scaled Fare should have std close to 1")
    
    def test_extract_title(self):
        """
        Test extraction of title from Name.
        """
        # Process data
        data_with_titles = self.preprocessor.extract_title(self.sample_data)
        
        # Check if Name column is preserved
        self.assertIn('Name', data_with_titles.columns, "Name column should be preserved")
        
        # Check if title dummy variables are created
        title_columns = [col for col in data_with_titles.columns if col.startswith('Title_')]
        self.assertTrue(len(title_columns) > 0, "Title dummy variables should be created")
        
        # Check specific titles
        expected_titles = ['Mr', 'Mrs', 'Miss']
        for title in expected_titles:
            title_col = f'Title_{title}'
            self.assertIn(title_col, data_with_titles.columns, f"{title_col} should be present")
    
    def test_create_family_features(self):
        """
        Test creation of family-related features.
        """
        # Process data
        data_with_family = self.preprocessor.create_family_features(self.sample_data)
        
        # Check if family size feature is created
        self.assertIn('FamilySize', data_with_family.columns, "FamilySize column should be created")
        
        # Check if IsAlone feature is created
        self.assertIn('IsAlone', data_with_family.columns, "IsAlone column should be created")
        
        # Check if FamilyGroup dummy variables are created
        family_group_cols = [col for col in data_with_family.columns if col.startswith('FamilyGroup_')]
        self.assertTrue(len(family_group_cols) > 0, "FamilyGroup dummy variables should be created")
        
        # Check values
        self.assertEqual(data_with_family.iloc[0]['FamilySize'], 2, 
                       "FamilySize should be SibSp + Parch + 1")
        
        # Sample with no siblings or parents should be marked as alone
        alone_samples = data_with_family[
            (data_with_family['SibSp'] == 0) & 
            (data_with_family['Parch'] == 0)
        ]
        for _, row in alone_samples.iterrows():
            self.assertEqual(row['IsAlone'], 1, "Passenger with no family should have IsAlone=1")
    
    def test_fit_transform(self):
        """
        Test complete fit_transform pipeline.
        """
        # Process data
        processed_data = self.preprocessor.fit_transform(self.sample_data)
        
        # Check shape
        self.assertTrue(processed_data.shape[0] == self.sample_data.shape[0], 
                      "Processed data should have same number of rows")
        
        # Check if unnecessary columns are dropped
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'is_train']
        for col in drop_cols:
            self.assertNotIn(col, processed_data.columns, f"{col} should be dropped")
        
        # Check if expected columns are present
        expected_patterns = ['Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived', 
                           'FamilySize', 'IsAlone', 'Embarked_', 'Pclass_', 'Title_', 'FamilyGroup_']
        
        for pattern in expected_patterns:
            matching_cols = [col for col in processed_data.columns if pattern in col]
            self.assertTrue(len(matching_cols) > 0, f"Columns matching '{pattern}' should be present")
        
        # Check if transformed_columns is stored
        self.assertIsNotNone(self.preprocessor.transformed_columns, 
                           "transformed_columns should be stored")
        self.assertEqual(len(self.preprocessor.transformed_columns), processed_data.shape[1], 
                       "transformed_columns should contain all column names")
    
    def test_transform(self):
        """
        Test transform with previously fitted preprocessor.
        """
        # First fit on the sample data
        self.preprocessor.fit_transform(self.sample_data)
        
        # Create new test data
        test_data = pd.DataFrame({
            'PassengerId': [6, 7],
            'Pclass': [3, 1],
            'Name': ['Moran, Mr. James', 'McCarthy, Miss. Catherine'],
            'Sex': ['male', 'female'],
            'Age': [np.nan, 30.0],
            'SibSp': [0, 0],
            'Parch': [0, 0],
            'Ticket': ['330877', '17598'],
            'Fare': [8.4583, 71.2833],
            'Cabin': [None, 'B4'],
            'Embarked': ['Q', 'C'],
            'is_train': [0, 0]
        })
        
        # Transform the test data
        transformed_test = self.preprocessor.transform(test_data)
        
        # Check if all expected columns are present
        self.assertEqual(set(transformed_test.columns), set(self.preprocessor.transformed_columns),
                       "Transformed test data should have same columns as fit_transform output")
        
        # Check shape
        self.assertEqual(transformed_test.shape[0], test_data.shape[0],
                       "Transformed data should have same number of rows as input")
    
    def test_transform_before_fit(self):
        """
        Test calling transform before fit_transform.
        """
        # Initialize a new preprocessor
        fresh_preprocessor = DataPreprocessor()
        
        # Call transform without prior fitting
        transformed_data = fresh_preprocessor.transform(self.sample_data)
        
        # This should run fit_transform internally, so check the output
        self.assertIsNotNone(transformed_data, "transform should handle case when fit_transform wasn't called")
        self.assertIsNotNone(fresh_preprocessor.transformed_columns, 
                           "transformed_columns should be set after automatic fit_transform")


if __name__ == '__main__':
    unittest.main()
