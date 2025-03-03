# -*- coding: utf-8 -*-
"""
Model Evaluator Module

This module is responsible for evaluating machine learning models on the Titanic dataset.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelEvaluator:
    """
    ModelEvaluator class for evaluating machine learning models.
    
    This class handles the evaluation of trained machine learning models
    using various performance metrics.
    """
    
    def __init__(self):
        """
        Initialize the ModelEvaluator.
        """
        pass
    
    def evaluate_model(self, model, X, y_true):
        """
        Evaluate a model using multiple metrics.
        
        Args:
            model: Trained machine learning model to evaluate.
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y_true (numpy.ndarray or pandas.Series): True target values.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = None
        
        # Some models don't have predict_proba
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
        except (AttributeError, IndexError):
            pass
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add AUC if probabilities are available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def compare_models(self, models_dict, X, y_true):
        """
        Compare multiple models using the same evaluation metrics.
        
        Args:
            models_dict (dict): Dictionary of trained models {model_name: model}.
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y_true (numpy.ndarray or pandas.Series): True target values.
            
        Returns:
            pandas.DataFrame: Comparison of models across evaluation metrics.
        """
        results = {}
        
        for model_name, model in models_dict.items():
            results[model_name] = self.evaluate_model(model, X, y_true)
        
        return pd.DataFrame(results).T
