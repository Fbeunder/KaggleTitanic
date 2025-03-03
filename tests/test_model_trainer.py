# -*- coding: utf-8 -*-
"""
Tests for Model Trainer Module
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from src.modelling.model_trainer import ModelTrainer
from src.modelling.model_factory import ModelFactory, TitanicModel


class TestModelTrainer(unittest.TestCase):
    """
    Test cases for the ModelTrainer class.
    """
    
    def setUp(self):
        """
        Set up test data and model trainer instance.
        """
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2, 
            random_state=42
        )
        
        # Convert to DataFrame for more realistic testing
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
        
        # Create simple model for testing
        self.model = LogisticRegression(random_state=42)
        self.titanic_model = ModelFactory.create_model('logistic_regression')
        
        # Create trainer
        self.trainer = ModelTrainer(random_state=42)
    
    def test_train(self):
        """
        Test the train method.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        trained_model = self.trainer.train(self.model, X_train, y_train, model_name="test_model")
        
        # Check that the model was trained
        self.assertIsNotNone(trained_model)
        
        # Check that the model was stored
        self.assertIn("test_model", self.trainer.trained_models)
        
        # Check that the model can make predictions
        y_pred = trained_model.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))
        
        # Train TitanicModel
        titanic_trained = self.trainer.train(self.titanic_model, X_train, y_train, model_name="titanic_model")
        self.assertIsNotNone(titanic_trained)
        self.assertIn("titanic_model", self.trainer.trained_models)
    
    def test_cross_validate(self):
        """
        Test the cross_validate method.
        """
        # Perform cross-validation
        result = self.trainer.cross_validate(
            self.model, self.X, self.y, cv=5, scoring='accuracy'
        )
        
        # Check result format
        self.assertIn('mean_test_score', result)
        self.assertIn('std_test_score', result)
        self.assertIn('cv_results', result)
        
        # Check result values
        self.assertGreater(result['mean_test_score'], 0)
        self.assertLess(result['mean_test_score'], 1)
        self.assertEqual(len(result['cv_results']), 5)
    
    def test_train_test_split_with_validation(self):
        """
        Test the train_test_split_with_validation method.
        """
        # Create DataFrame with target column
        data = self.X.copy()
        data['target'] = self.y
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.trainer.train_test_split_with_validation(
                data, 'target', test_size=0.2, val_size=0.2, stratify=True
            )
        
        # Check shapes
        expected_test_samples = int(len(data) * 0.2)
        expected_val_samples = int((len(data) - expected_test_samples) * 0.2)
        expected_train_samples = len(data) - expected_test_samples - expected_val_samples
        
        self.assertEqual(len(X_test), expected_test_samples)
        self.assertEqual(len(X_val), expected_val_samples)
        self.assertEqual(len(X_train), expected_train_samples)
        
        # Check that columns are preserved
        self.assertEqual(set(X_train.columns), set(self.X.columns))
        self.assertEqual(set(X_val.columns), set(self.X.columns))
        self.assertEqual(set(X_test.columns), set(self.X.columns))
    
    def test_get_best_hyperparameters(self):
        """
        Test the get_best_hyperparameters method.
        """
        # Define simple param grid for faster tests
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l2']
        }
        
        # Find best parameters
        result = self.trainer.get_best_hyperparameters(
            'logistic_regression', param_grid, self.X, self.y, cv=3
        )
        
        # Check result format
        self.assertIn('best_params', result)
        self.assertIn('best_score', result)
        self.assertIn('cv_results', result)
        
        # Check that best_params contains expected keys
        self.assertIn('C', result['best_params'])
        self.assertIn('penalty', result['best_params'])
    
    def test_train_multiple_models(self):
        """
        Test the train_multiple_models method.
        """
        # Create models
        models = {
            'logistic': LogisticRegression(random_state=42),
            'titanic_logistic': ModelFactory.create_model('logistic_regression')
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train models
        result = self.trainer.train_multiple_models(
            models, X_train, y_train, X_val=X_test, y_val=y_test
        )
        
        # Check result format
        self.assertIn('trained_models', result)
        self.assertIn('performance', result)
        
        # Check that all models were trained
        for model_name in models:
            self.assertIn(model_name, result['trained_models'])
            self.assertIn(model_name, result['performance'])
        
        # Check performance values
        for model_name, perf in result['performance'].items():
            self.assertGreaterEqual(perf, 0)
            self.assertLessEqual(perf, 1)


if __name__ == '__main__':
    unittest.main()
