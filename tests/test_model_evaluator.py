# -*- coding: utf-8 -*-
"""
Tests for Model Evaluator Module
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from src.modelling.model_evaluator import ModelEvaluator
from src.modelling.model_factory import ModelFactory


class TestModelEvaluator(unittest.TestCase):
    """
    Test cases for the ModelEvaluator class.
    """
    
    def setUp(self):
        """
        Set up test data and model evaluator instance.
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
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Create and train models
        self.sklearn_model = LogisticRegression(random_state=42)
        self.sklearn_model.fit(self.X_train, self.y_train)
        
        self.titanic_model = ModelFactory.create_model('logistic_regression')
        self.titanic_model.fit(self.X_train, self.y_train)
        
        # Create evaluator
        self.evaluator = ModelEvaluator()
        
        # Make predictions for testing
        self.y_pred = self.sklearn_model.predict(self.X_test)
        self.y_pred_proba = self.sklearn_model.predict_proba(self.X_test)[:, 1]
    
    def test_evaluate_model(self):
        """
        Test the evaluate_model method.
        """
        # Evaluate sklearn model
        sklearn_metrics = self.evaluator.evaluate_model(
            self.sklearn_model, self.X_test, self.y_test, model_name="sklearn_model"
        )
        
        # Check metrics format
        self.assertIn('accuracy', sklearn_metrics)
        self.assertIn('precision', sklearn_metrics)
        self.assertIn('recall', sklearn_metrics)
        self.assertIn('f1', sklearn_metrics)
        self.assertIn('roc_auc', sklearn_metrics)
        self.assertIn('confusion_matrix', sklearn_metrics)
        self.assertIn('classification_report', sklearn_metrics)
        
        # Check that model was stored in evaluation results
        self.assertIn("sklearn_model", self.evaluator.evaluation_results)
        
        # Evaluate Titanic model
        titanic_metrics = self.evaluator.evaluate_model(
            self.titanic_model, self.X_test, self.y_test, model_name="titanic_model"
        )
        
        # Check that model was stored in evaluation results
        self.assertIn("titanic_model", self.evaluator.evaluation_results)
    
    def test_calculate_metrics(self):
        """
        Test the calculate_metrics method.
        """
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(
            self.y_test, self.y_pred, self.y_pred_proba
        )
        
        # Check metrics format
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('classification_report', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['precision'], 0)
        self.assertLessEqual(metrics['precision'], 1)
        self.assertGreaterEqual(metrics['recall'], 0)
        self.assertLessEqual(metrics['recall'], 1)
        self.assertGreaterEqual(metrics['f1'], 0)
        self.assertLessEqual(metrics['f1'], 1)
        self.assertGreaterEqual(metrics['roc_auc'], 0)
        self.assertLessEqual(metrics['roc_auc'], 1)
        
        # Test without probabilities
        metrics_no_proba = self.evaluator.calculate_metrics(self.y_test, self.y_pred)
        self.assertNotIn('roc_auc', metrics_no_proba)
    
    def test_get_confusion_matrix(self):
        """
        Test the get_confusion_matrix method.
        """
        # Get confusion matrix
        cm = self.evaluator.get_confusion_matrix(self.y_test, self.y_pred)
        
        # Check shape (binary classification = 2x2 matrix)
        self.assertEqual(cm.shape, (2, 2))
        
        # Check that values are non-negative integers
        self.assertTrue(np.all(cm >= 0))
        self.assertTrue(np.all(cm.astype(int) == cm))
        
        # Test normalized confusion matrix
        cm_norm = self.evaluator.get_confusion_matrix(
            self.y_test, self.y_pred, normalize='true'
        )
        
        # Check shape
        self.assertEqual(cm_norm.shape, (2, 2))
        
        # Check that values are between 0 and 1
        self.assertTrue(np.all(cm_norm >= 0))
        self.assertTrue(np.all(cm_norm <= 1))
        
        # Check that rows sum to 1 (normalize='true')
        self.assertTrue(np.allclose(cm_norm.sum(axis=1), np.ones(2)))
    
    def test_plot_confusion_matrix(self):
        """
        Test the plot_confusion_matrix method.
        """
        # Plot confusion matrix
        fig = self.evaluator.plot_confusion_matrix(
            self.y_test, self.y_pred, title="Test Confusion Matrix"
        )
        
        # Check that a figure was returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the title was set
        self.assertEqual(fig.axes[0].get_title(), "Test Confusion Matrix")
        
        # Test with model name from stored results
        # First evaluate a model to store results
        self.evaluator.evaluate_model(
            self.sklearn_model, self.X_test, self.y_test, model_name="test_model"
        )
        
        # Then plot using model name
        fig = self.evaluator.plot_confusion_matrix(model_name="test_model")
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with normalization
        fig = self.evaluator.plot_confusion_matrix(
            self.y_test, self.y_pred, normalize='true'
        )
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_roc_curve(self):
        """
        Test the plot_roc_curve method.
        """
        # Plot ROC curve
        fig = self.evaluator.plot_roc_curve(
            self.y_test, self.y_pred_proba, title="Test ROC Curve"
        )
        
        # Check that a figure was returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the title was set
        self.assertEqual(fig.axes[0].get_title(), "Test ROC Curve")
        
        # Test with model name from stored results
        # First evaluate a model to store results
        self.evaluator.evaluate_model(
            self.sklearn_model, self.X_test, self.y_test, model_name="test_model"
        )
        
        # Then plot using model name
        fig = self.evaluator.plot_roc_curve(model_name="test_model")
        self.assertIsInstance(fig, plt.Figure)
    
    def test_compare_models(self):
        """
        Test the compare_models method.
        """
        # Create models to compare
        models = {
            'model1': self.sklearn_model,
            'model2': self.titanic_model
        }
        
        # Compare models
        comparison_df = self.evaluator.compare_models(models, self.X_test, self.y_test)
        
        # Check that the result is a DataFrame
        self.assertIsInstance(comparison_df, pd.DataFrame)
        
        # Check that all models are included
        for model_name in models:
            self.assertIn(model_name, comparison_df.index)
        
        # Check that metrics are included
        self.assertIn('accuracy', comparison_df.columns)
        self.assertIn('precision', comparison_df.columns)
        self.assertIn('recall', comparison_df.columns)
        self.assertIn('f1', comparison_df.columns)
        
        # Test comparing previously evaluated models
        # First, clear any existing evaluation results
        self.evaluator.evaluation_results = {}
        
        # Evaluate models individually
        for model_name, model in models.items():
            self.evaluator.evaluate_model(model, self.X_test, self.y_test, model_name=model_name)
        
        # Compare using stored results
        comparison_df = self.evaluator.compare_models()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(comparison_df, pd.DataFrame)
        
        # Check that all models are included
        for model_name in models:
            self.assertIn(model_name, comparison_df.index)
    
    def test_plot_model_comparison(self):
        """
        Test the plot_model_comparison method.
        """
        # Create models to compare
        models = {
            'model1': self.sklearn_model,
            'model2': self.titanic_model
        }
        
        # Plot model comparison
        fig = self.evaluator.plot_model_comparison(
            models, self.X_test, self.y_test, metrics=['accuracy', 'precision', 'recall']
        )
        
        # Check that a figure was returned
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with previously evaluated models
        # First, clear any existing evaluation results
        self.evaluator.evaluation_results = {}
        
        # Evaluate models individually
        for model_name, model in models.items():
            self.evaluator.evaluate_model(model, self.X_test, self.y_test, model_name=model_name)
        
        # Plot comparison using stored results
        fig = self.evaluator.plot_model_comparison()
        self.assertIsInstance(fig, plt.Figure)


if __name__ == '__main__':
    unittest.main()
