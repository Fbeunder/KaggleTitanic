# -*- coding: utf-8 -*-
"""
Model Factory Module

This module implements the factory pattern for creating different machine learning models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class ModelFactory:
    """
    Factory class for creating different machine learning models.
    
    This class provides a centralized way to create and configure
    machine learning models for the Titanic survival prediction project.
    """
    
    @staticmethod
    def get_model(model_name, **kwargs):
        """
        Create and return a machine learning model based on the model name.
        
        Args:
            model_name (str): Name of the model to create.
            **kwargs: Additional parameters to pass to the model constructor.
            
        Returns:
            object: An instance of the requested machine learning model.
            
        Raises:
            ValueError: If the requested model is not supported.
        """
        models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svc': SVC,
            'decision_tree': DecisionTreeClassifier,
            'knn': KNeighborsClassifier
        }
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not supported. Available models: {list(models.keys())}")
        
        return models[model_name](**kwargs)
    
    @staticmethod
    def get_available_models():
        """
        Get a list of available model names.
        
        Returns:
            list: List of available model names.
        """
        return [
            'logistic_regression',
            'random_forest',
            'svc',
            'decision_tree',
            'knn'
        ]
