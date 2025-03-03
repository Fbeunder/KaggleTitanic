# -*- coding: utf-8 -*-
"""
Model Trainer Module

This module is responsible for training machine learning models on the Titanic dataset.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split


class ModelTrainer:
    """
    ModelTrainer class for training machine learning models.
    
    This class handles the training process for machine learning models
    on the Titanic dataset, including cross-validation.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state (int, optional): Random state for reproducibility.
        """
        self.random_state = random_state
        self.trained_models = {}
    
    def train_model(self, model, X, y, model_name=None):
        """
        Train a model on the provided data.
        
        Args:
            model: Machine learning model to train.
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y (numpy.ndarray or pandas.Series): Target vector.
            model_name (str, optional): Name to assign to the trained model.
                                       If None, a default name is generated.
                                       
        Returns:
            object: The trained model.
        """
        # Train the model
        model.fit(X, y)
        
        # Store the trained model if a name is provided
        if model_name:
            self.trained_models[model_name] = model
        
        return model
    
    def cross_validate(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Machine learning model to evaluate.
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y (numpy.ndarray or pandas.Series): Target vector.
            cv (int, optional): Number of cross-validation folds.
            scoring (str, optional): Scoring metric to use.
            
        Returns:
            tuple: Mean score and array of scores from cross-validation.
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores.mean(), scores
