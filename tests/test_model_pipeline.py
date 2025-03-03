# -*- coding: utf-8 -*-
"""
End-to-End Tests for the Model Training and Evaluation Pipeline
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer
from src.modelling.model_evaluator import ModelEvaluator


class TestModelPipeline(unittest.TestCase):
    """
    End-to-End test cases for the complete model training and evaluation pipeline.
    """
    
    def setUp(self):
        """
        Set up test data and instances.
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
        
        # Create DataFrame with target column
        self.data = self.X.copy()
        self.data['target'] = self.y
        
        # Create instances
        self.model_factory = ModelFactory
        self.trainer = ModelTrainer(random_state=42)
        self.evaluator = ModelEvaluator()
    
    def test_end_to_end_pipeline(self):
        """
        Test the complete model training and evaluation pipeline.
        """
        # Split data into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.trainer.train_test_split_with_validation(
                self.data, 'target', test_size=0.2, val_size=0.2
            )
        
        # Create multiple models
        models = {
            'logistic': self.model_factory.create_model('logistic_regression'),
            'random_forest': self.model_factory.create_model('random_forest', n_estimators=50),
            'svm': self.model_factory.create_model('svm')
        }
        
        # Train all models
        training_result = self.trainer.train_multiple_models(
            models, X_train, y_train, X_val, y_val
        )
        
        # Check that all models were trained
        self.assertEqual(len(training_result['trained_models']), len(models))
        self.assertEqual(len(training_result['performance']), len(models))
        
        # Get the best model based on validation performance
        best_model_name = max(
            training_result['performance'].items(), 
            key=lambda x: x[1]
        )[0]
        best_model = training_result['trained_models'][best_model_name]
        
        # Evaluate the best model on test data
        test_metrics = self.evaluator.evaluate_model(
            best_model, X_test, y_test, model_name=f"{best_model_name}_test"
        )
        
        # Check key metrics
        self.assertIn('accuracy', test_metrics)
        self.assertIn('precision', test_metrics)
        self.assertIn('recall', test_metrics)
        self.assertIn('f1', test_metrics)
        
        # Test hyperparameter tuning on best model type
        param_grid = {
            'n_estimators': [50, 100] if best_model_name == 'random_forest' else None,
            'C': [0.1, 1.0] if best_model_name in ['logistic', 'svm'] else None
        }
        
        # Only perform tuning if we have a parameter grid for this model type
        if any(param_grid.values()):
            tuning_result = self.trainer.get_best_hyperparameters(
                best_model_name.replace('logistic', 'logistic_regression'),
                {k: v for k, v in param_grid.items() if v is not None},
                X_train, y_train, cv=3
            )
            
            # Check tuning results
            self.assertIn('best_params', tuning_result)
            self.assertIn('best_score', tuning_result)
            
            # Create tuned model
            tuned_model = self.model_factory.create_model(
                best_model_name.replace('logistic', 'logistic_regression'),
                **tuning_result['best_params']
            )
            
            # Train tuned model
            self.trainer.train(tuned_model, X_train, y_train, model_name="tuned_model")
            
            # Evaluate tuned model
            tuned_metrics = self.evaluator.evaluate_model(
                tuned_model, X_test, y_test, model_name="tuned_model"
            )
            
            # Compare model performance
            performance_comparison = self.evaluator.compare_models(
                metric='accuracy'
            )
            
            # Check that the comparison includes all evaluated models
            self.assertIn(f"{best_model_name}_test", performance_comparison.index)
            self.assertIn("tuned_model", performance_comparison.index)
            
            # Generate plots (test that they run without errors)
            try:
                # Confusion matrix
                cm_fig = self.evaluator.plot_confusion_matrix(model_name="tuned_model")
                
                # ROC curve
                roc_fig = self.evaluator.plot_roc_curve(model_name="tuned_model")
                
                # Model comparison
                comparison_fig = self.evaluator.plot_model_comparison()
            except Exception as e:
                self.fail(f"Visualization failed with error: {e}")


if __name__ == '__main__':
    unittest.main()
