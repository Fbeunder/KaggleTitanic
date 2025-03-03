# -*- coding: utf-8 -*-
"""
Tests for Feature Engineering modules

This module contains tests for the feature_creator.py and feature_selector.py modules.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering.feature_creator import FeatureCreator
from src.feature_engineering.feature_selector import FeatureSelector


class TestFeatureCreator(unittest.TestCase):
    """
    Test cases for FeatureCreator class.
    """
    
    def setUp(self):
        """
        Set up test data for FeatureCreator tests.
        """
        # Create a sample Titanic-like dataset
        self.data = pd.DataFrame({
            'PassengerId': range(1, 11),
            'Survived': [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
            'Pclass': [3, 1, 1, 2, 3, 3, 1, 2, 3, 1],
            'Name': [
                'Braund, Mr. Owen Harris',
                'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                'Allen, Mr. William Henry',
                'Moran, Mr. James',
                'McCarthy, Mr. Timothy J',
                'Palsson, Master. Gosta Leonard',
                'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
                'Nasser, Mrs. Nicholas (Adele Achem)'
            ],
            'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
            'Age': [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14],
            'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
            'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450', '330877', '17463', '349909', '347742', '237736'],
            'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708],
            'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, 'C7'],
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C']
        })
        
        # Initialize FeatureCreator
        self.feature_creator = FeatureCreator()
    
    def test_create_title_feature(self):
        """
        Test the create_title_feature method.
        """
        # Apply title feature creation
        result = self.feature_creator.create_title_feature(self.data)
        
        # Check that Title column was created
        self.assertIn('Title', result.columns)
        
        # Check that title one-hot encoded columns were created
        self.assertTrue(any(col.startswith('Title_') for col in result.columns))
        
        # Check specific titles for a few examples
        titles = result['Title'].unique()
        self.assertIn('Mr', titles)
        self.assertIn('Mrs', titles)
        self.assertIn('Miss', titles)
        self.assertIn('Master', titles)
    
    def test_create_family_size_feature(self):
        """
        Test the create_family_size_feature method.
        """
        # Apply family size feature creation
        result = self.feature_creator.create_family_size_feature(self.data)
        
        # Check that FamilySize column was created
        self.assertIn('FamilySize', result.columns)
        
        # Check that IsAlone column was created
        self.assertIn('IsAlone', result.columns)
        
        # Check that FamilyType column was created
        self.assertIn('FamilyType', result.columns)
        
        # Check FamilySize calculation for a specific passenger
        # Passenger with PassengerId 8 has SibSp=3, Parch=1, so FamilySize should be 5
        self.assertEqual(result.loc[7, 'FamilySize'], 5)
        
        # Check IsAlone for a passenger traveling alone
        # Passenger with PassengerId 5 has SibSp=0, Parch=0, so IsAlone should be 1
        self.assertEqual(result.loc[4, 'IsAlone'], 1)
    
    def test_create_cabin_features(self):
        """
        Test the create_cabin_features method.
        """
        # Apply cabin feature creation
        result = self.feature_creator.create_cabin_features(self.data)
        
        # Check that HasCabin column was created
        self.assertIn('HasCabin', result.columns)
        
        # Check that CabinDeck column was created
        self.assertIn('CabinDeck', result.columns)
        
        # Check HasCabin value for a passenger with cabin information
        # Passenger with PassengerId 2 has Cabin='C85', so HasCabin should be 1
        self.assertEqual(result.loc[1, 'HasCabin'], 1)
        
        # Check HasCabin value for a passenger without cabin information
        # Passenger with PassengerId 1 has Cabin=NaN, so HasCabin should be 0
        self.assertEqual(result.loc[0, 'HasCabin'], 0)
        
        # Check CabinDeck value for a passenger with cabin information
        # Passenger with PassengerId 2 has Cabin='C85', so CabinDeck should be 'C'
        self.assertEqual(result.loc[1, 'CabinDeck'], 'C')
    
    def test_create_age_categories(self):
        """
        Test the create_age_categories method.
        """
        # Apply age category feature creation
        result = self.feature_creator.create_age_categories(self.data)
        
        # Check that AgeCategory column was created
        self.assertIn('AgeCategory', result.columns)
        
        # Check that IsChild column was created
        self.assertIn('IsChild', result.columns)
        
        # Check AgeCategory for a child
        # Passenger with PassengerId 8 has Age=2, so AgeCategory should be 'Infant'
        self.assertEqual(result.loc[7, 'AgeCategory'], 'Infant')
        
        # Check IsChild value for a child
        # Passenger with PassengerId 8 has Age=2, so IsChild should be 1
        self.assertEqual(result.loc[7, 'IsChild'], 1)
        
        # Check IsChild value for an adult
        # Passenger with PassengerId 1 has Age=22, so IsChild should be 0
        self.assertEqual(result.loc[0, 'IsChild'], 0)
    
    def test_create_fare_categories(self):
        """
        Test the create_fare_categories method.
        """
        # Apply fare category feature creation
        result = self.feature_creator.create_fare_categories(self.data)
        
        # Check that FareCategory column was created
        self.assertIn('FareCategory', result.columns)
        
        # Check that LogFare column was created
        self.assertIn('LogFare', result.columns)
        
        # Check number of fare categories (should be 5)
        self.assertEqual(len(result['FareCategory'].unique()), 5)
    
    def test_create_all_features(self):
        """
        Test the create_all_features method.
        """
        # Apply all feature creation
        result = self.feature_creator.create_all_features(self.data)
        
        # Check that original columns are preserved
        for col in self.data.columns:
            self.assertIn(col, result.columns)
        
        # Check that new feature columns are created
        new_features = ['Title', 'FamilySize', 'IsAlone', 'HasCabin', 'AgeCategory', 'IsChild', 'FareCategory', 'LogFare']
        for feature in new_features:
            self.assertTrue(
                feature in result.columns or any(col.startswith(f'{feature}_') for col in result.columns),
                f"Expected feature {feature} or its one-hot encoded version not found"
            )
        
        # Check that the number of columns increased
        self.assertGreater(len(result.columns), len(self.data.columns))


class TestFeatureSelector(unittest.TestCase):
    """
    Test cases for FeatureSelector class.
    """
    
    def setUp(self):
        """
        Set up test data for FeatureSelector tests.
        """
        # Create synthetic classification dataset
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.X = pd.DataFrame(X, columns=feature_names)
        self.y = pd.Series(y, name='target')
        
        # Initialize FeatureSelector
        self.feature_selector = FeatureSelector()
    
    def test_select_k_best(self):
        """
        Test the select_k_best method.
        """
        # Select top 5 features
        X_selected = self.feature_selector.select_k_best(self.X, self.y, k=5)
        
        # Check that the number of columns is reduced to 5
        self.assertEqual(X_selected.shape[1], 5)
        
        # Check that selected_features attribute is set
        self.assertIsNotNone(self.feature_selector.selected_features)
        self.assertEqual(len(self.feature_selector.selected_features), 5)
        
        # Check that feature_importance attribute is set
        self.assertIsNotNone(self.feature_selector.feature_importance)
    
    def test_select_with_rfe(self):
        """
        Test the select_with_rfe method.
        """
        # Select top 5 features using RFE
        X_selected = self.feature_selector.select_with_rfe(self.X, self.y, n_features_to_select=5)
        
        # Check that the number of columns is reduced to 5
        self.assertEqual(X_selected.shape[1], 5)
        
        # Check that selected_features attribute is set
        self.assertIsNotNone(self.feature_selector.selected_features)
        self.assertEqual(len(self.feature_selector.selected_features), 5)
    
    def test_get_feature_importance(self):
        """
        Test the get_feature_importance method.
        """
        # Get feature importance using random forest
        importance_df = self.feature_selector.get_feature_importance(self.X, self.y, method='random_forest')
        
        # Check that importance_df is a DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame)
        
        # Check that importance_df has Feature and Importance columns
        self.assertIn('Feature', importance_df.columns)
        self.assertIn('Importance', importance_df.columns)
        
        # Check that feature_importance attribute is set
        self.assertIsNotNone(self.feature_selector.feature_importance)
        
        # Check that the number of rows equals the number of features
        self.assertEqual(len(importance_df), self.X.shape[1])
    
    def test_remove_highly_correlated(self):
        """
        Test the remove_highly_correlated method.
        """
        # Create DataFrame with highly correlated features
        X_corr = self.X.copy()
        X_corr['feature_20'] = X_corr['feature_0'] * 0.9 + np.random.normal(0, 0.1, size=len(X_corr))
        X_corr['feature_21'] = X_corr['feature_1'] * 0.95 + np.random.normal(0, 0.05, size=len(X_corr))
        
        # Remove highly correlated features
        X_filtered = self.feature_selector.remove_highly_correlated(X_corr, threshold=0.8)
        
        # Check that the number of columns is reduced
        self.assertLess(X_filtered.shape[1], X_corr.shape[1])


if __name__ == '__main__':
    unittest.main()
