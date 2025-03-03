# -*- coding: utf-8 -*-
"""
Unit tests for the ModelFactory module.

This module contains tests for the ModelFactory and model implementations.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.modelling.model_factory import (
    ModelFactory, TitanicModel, LogisticRegressionModel, RandomForestModel,
    DecisionTreeModel, SVMModel, KNeighborsModel, GradientBoostingModel
)


class MockModel(TitanicModel):
    """Mock model implementation for testing."""
    
    def get_default_params(self):
        return {'mock_param': 'test'}
    
    def _initialize_model(self):
        # Simple mock model that uses a LogisticRegression under the hood
        self.model = LogisticRegression()


class TestModelFactory(unittest.TestCase):
    """
    Test cases for the ModelFactory class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create synthetic data for testing
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_get_available_models(self):
        """
        Test getting available models.
        """
        available_models = ModelFactory.get_available_models()
        expected_models = [
            'logistic_regression', 'random_forest', 'decision_tree',
            'svm', 'knn', 'gradient_boosting'
        ]
        
        # Check that all expected models are available
        for model_type in expected_models:
            self.assertIn(model_type, available_models)
    
    def test_create_model(self):
        """
        Test creating models of different types.
        """
        # Test creating each model type
        model_types = ModelFactory.get_available_models()
        
        for model_type in model_types:
            model = ModelFactory.create_model(model_type)
            self.assertIsInstance(model, TitanicModel)
            self.assertTrue(hasattr(model, 'fit'))
            self.assertTrue(hasattr(model, 'predict'))
            self.assertTrue(hasattr(model, 'predict_proba'))
            self.assertTrue(hasattr(model, 'evaluate'))
    
    def test_create_invalid_model(self):
        """
        Test creating an invalid model type.
        """
        with self.assertRaises(ValueError):
            ModelFactory.create_model('invalid_model_type')
    
    def test_model_default_params(self):
        """
        Test getting default parameters for models.
        """
        model_types = ModelFactory.get_available_models()
        
        for model_type in model_types:
            params = ModelFactory.get_model_default_params(model_type)
            self.assertIsInstance(params, dict)
            self.assertTrue(len(params) > 0)
    
    def test_get_invalid_model_params(self):
        """
        Test getting parameters for an invalid model type.
        """
        with self.assertRaises(ValueError):
            ModelFactory.get_model_default_params('invalid_model_type')
    
    def test_register_model(self):
        """
        Test registering a new model type.
        """
        # Register the mock model
        ModelFactory.register_model('mock_model', MockModel)
        
        # Check that it was added to available models
        available_models = ModelFactory.get_available_models()
        self.assertIn('mock_model', available_models)
        
        # Test creating the model
        mock_model = ModelFactory.create_model('mock_model')
        self.assertIsInstance(mock_model, MockModel)
    
    def test_register_invalid_model(self):
        """
        Test registering an invalid model class.
        """
        # Try to register a class that doesn't inherit from TitanicModel
        with self.assertRaises(TypeError):
            ModelFactory.register_model('invalid', dict)


class TestModelImplementations(unittest.TestCase):
    """
    Test cases for the specific model implementations.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create synthetic data for testing
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
    
    def test_logistic_regression_model(self):
        """
        Test LogisticRegressionModel.
        """
        model = LogisticRegressionModel()
        self.assertIsInstance(model.model, LogisticRegression)
        
        # Test with custom parameters
        custom_model = LogisticRegressionModel(C=0.5, penalty='l1')
        self.assertEqual(custom_model.params['C'], 0.5)
        self.assertEqual(custom_model.params['penalty'], 'l1')
    
    def test_random_forest_model(self):
        """
        Test RandomForestModel.
        """
        model = RandomForestModel()
        self.assertIsInstance(model.model, RandomForestClassifier)
        
        # Test with custom parameters
        custom_model = RandomForestModel(n_estimators=50, max_depth=5)
        self.assertEqual(custom_model.params['n_estimators'], 50)
        self.assertEqual(custom_model.params['max_depth'], 5)
    
    def test_decision_tree_model(self):
        """
        Test DecisionTreeModel.
        """
        model = DecisionTreeModel()
        self.assertIsInstance(model.model, DecisionTreeClassifier)
        
        # Test with custom parameters
        custom_model = DecisionTreeModel(max_depth=10)
        self.assertEqual(custom_model.params['max_depth'], 10)
    
    def test_svm_model(self):
        """
        Test SVMModel.
        """
        model = SVMModel()
        self.assertIsInstance(model.model, SVC)
        
        # Test with custom parameters
        custom_model = SVMModel(C=10.0, kernel='linear')
        self.assertEqual(custom_model.params['C'], 10.0)
        self.assertEqual(custom_model.params['kernel'], 'linear')
    
    def test_knn_model(self):
        """
        Test KNeighborsModel.
        """
        model = KNeighborsModel()
        self.assertIsInstance(model.model, KNeighborsClassifier)
        
        # Test with custom parameters
        custom_model = KNeighborsModel(n_neighbors=10, weights='distance')
        self.assertEqual(custom_model.params['n_neighbors'], 10)
        self.assertEqual(custom_model.params['weights'], 'distance')
    
    def test_gradient_boosting_model(self):
        """
        Test GradientBoostingModel.
        """
        model = GradientBoostingModel()
        self.assertIsInstance(model.model, GradientBoostingClassifier)
        
        # Test with custom parameters
        custom_model = GradientBoostingModel(n_estimators=200, learning_rate=0.05)
        self.assertEqual(custom_model.params['n_estimators'], 200)
        self.assertEqual(custom_model.params['learning_rate'], 0.05)


class TestModelFunctionality(unittest.TestCase):
    """
    Test cases for the model functionality (fit, predict, evaluate).
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create synthetic data for testing
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Use a simple model for testing
        self.model = LogisticRegressionModel()
    
    def test_fit_and_predict(self):
        """
        Test fitting a model and making predictions.
        """
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Check predictions shape and type
        self.assertEqual(len(y_pred), len(self.X_test))
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))
    
    def test_predict_proba(self):
        """
        Test probability predictions.
        """
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # Get probability predictions
        y_proba = self.model.predict_proba(self.X_test)
        
        # Check shape and values
        self.assertEqual(y_proba.shape, (len(self.X_test), 2))
        self.assertTrue(np.all(y_proba >= 0))
        self.assertTrue(np.all(y_proba <= 1))
        self.assertTrue(np.allclose(np.sum(y_proba, axis=1), 1.0))
    
    def test_evaluate(self):
        """
        Test model evaluation.
        """
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate the model
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Check that metrics contains expected keys
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)
    
    def test_feature_importance(self):
        """
        Test getting feature importance.
        """
        # For this test, use RandomForest which has feature_importances_
        model = RandomForestModel()
        model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        importance_df = model.get_feature_importance()
        
        # Check result
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertEqual(len(importance_df), self.X_train.shape[1])
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
        # Test with feature names
        feature_names = self.X_train.columns
        importance_df = model.get_feature_importance(feature_names)
        self.assertTrue(set(importance_df['feature']) == set(feature_names))
        
        # Also test a model with coefficients instead of feature_importances_
        coef_model = LogisticRegressionModel()
        coef_model.fit(self.X_train, self.y_train)
        coef_df = coef_model.get_feature_importance(feature_names)
        self.assertIsInstance(coef_df, pd.DataFrame)
        self.assertEqual(len(coef_df), self.X_train.shape[1])
        
        # Test model without feature importance
        mock_model = MockModel()
        mock_model.model.fit = lambda X, y: None  # Mock fit method
        mock_model.fit(self.X_train, self.y_train)
        # Remove feature_importances_ and coef_ if present
        if hasattr(mock_model.model, 'feature_importances_'):
            delattr(mock_model.model, 'feature_importances_')
        if hasattr(mock_model.model, 'coef_'):
            delattr(mock_model.model, 'coef_')
        result = mock_model.get_feature_importance()
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
