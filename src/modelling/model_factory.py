# -*- coding: utf-8 -*-
"""
Model Factory Module

This module implements the factory pattern for creating different machine learning models
for Titanic survival prediction. It provides a standardized interface for all models,
default parameters, and easy model instantiation.
"""

import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TitanicModel(ABC):
    """
    Abstract base class for all Titanic prediction models.
    
    This class defines the interface that all model implementations must follow,
    ensuring a standardized way to interact with different models.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model with custom parameters.
        
        Args:
            **kwargs: Model-specific parameters that override defaults.
        """
        self.name = self.__class__.__name__
        self.model = None
        self.params = self.get_default_params()
        self.params.update(kwargs)
        self._initialize_model()
        
    @abstractmethod
    def get_default_params(self):
        """
        Get the default parameters for this model.
        
        Returns:
            dict: Default parameters for model initialization.
        """
        pass
    
    def _initialize_model(self):
        """
        Initialize the underlying model with current parameters.
        """
        pass
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X (array-like): Training features.
            y (array-like): Target values.
            
        Returns:
            self: The fitted model.
        """
        logger.info(f"Training {self.name} with {X.shape[0]} samples and {X.shape[1]} features")
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (array-like): Features to predict.
            
        Returns:
            array: Predicted class labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        
        Args:
            X (array-like): Features to predict.
            
        Returns:
            array: Probability estimates. 
                  Returns shape (n_samples, n_classes).
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models that don't have predict_proba
            logger.warning(f"{self.name} doesn't support probability prediction, returning one-hot encoding")
            y_pred = self.predict(X)
            return np.column_stack((1-y_pred, y_pred))
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Args:
            X (array-like): Test features.
            y (array-like): True target values.
            
        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if y_proba.shape[1] > 1 else self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        logger.info(f"Evaluation metrics for {self.name}: {metrics}")
        return metrics
    
    def tune_hyperparameters(self, X, y, param_grid=None, cv=5, scoring='accuracy'):
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X (array-like): Training features.
            y (array-like): Target values.
            param_grid (dict, optional): Grid of parameters to search. 
                                        If None, uses a default grid.
            cv (int, optional): Number of cross-validation folds.
            scoring (str, optional): Scoring metric to optimize.
            
        Returns:
            self: The model with optimized parameters.
        """
        if param_grid is None:
            param_grid = self.get_param_grid()
            
        if not param_grid:
            logger.warning(f"No parameter grid defined for {self.name}. Skipping hyperparameter tuning.")
            return self
            
        logger.info(f"Tuning hyperparameters for {self.name} with {len(param_grid)} parameter combinations")
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters found for {self.name}: {grid_search.best_params_}")
        logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.params.update(grid_search.best_params_)
        self._initialize_model()
        self.model.fit(X, y)
        
        return self
    
    def get_param_grid(self):
        """
        Get the parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {}
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance if supported by the model.
        
        Args:
            feature_names (list, optional): Names of features to use in the result.
            
        Returns:
            pandas.DataFrame or None: Feature importance dataframe if supported,
                                     None otherwise.
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(coef))]
                
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coef
            }).sort_values('coefficient', key=abs, ascending=False)
            
            return importance_df
        
        else:
            logger.warning(f"{self.name} doesn't support feature importance extraction")
            return None


class LogisticRegressionModel(TitanicModel):
    """
    Logistic Regression model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """
        Get default parameters for Logistic Regression.
        
        Returns:
            dict: Default parameters.
        """
        return {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'random_state': 42,
            'max_iter': 1000
        }
    
    def _initialize_model(self):
        """Initialize the Logistic Regression model."""
        self.model = LogisticRegression(**self.params)
    
    def get_param_grid(self):
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }


class RandomForestModel(TitanicModel):
    """
    Random Forest model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """
        Get default parameters for Random Forest.
        
        Returns:
            dict: Default parameters.
        """
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def _initialize_model(self):
        """Initialize the Random Forest model."""
        self.model = RandomForestClassifier(**self.params)
    
    def get_param_grid(self):
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


class DecisionTreeModel(TitanicModel):
    """
    Decision Tree model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """
        Get default parameters for Decision Tree.
        
        Returns:
            dict: Default parameters.
        """
        return {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def _initialize_model(self):
        """Initialize the Decision Tree model."""
        self.model = DecisionTreeClassifier(**self.params)
    
    def get_param_grid(self):
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


class SVMModel(TitanicModel):
    """
    Support Vector Machine model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """
        Get default parameters for SVM.
        
        Returns:
            dict: Default parameters.
        """
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    def _initialize_model(self):
        """Initialize the SVM model."""
        self.model = SVC(**self.params)
    
    def get_param_grid(self):
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }


class KNeighborsModel(TitanicModel):
    """
    K-Nearest Neighbors model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """
        Get default parameters for KNN.
        
        Returns:
            dict: Default parameters.
        """
        return {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'p': 2  # Euclidean distance
        }
    
    def _initialize_model(self):
        """Initialize the KNN model."""
        self.model = KNeighborsClassifier(**self.params)
    
    def get_param_grid(self):
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1: Manhattan, 2: Euclidean
        }


class GradientBoostingModel(TitanicModel):
    """
    Gradient Boosting model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """
        Get default parameters for Gradient Boosting.
        
        Returns:
            dict: Default parameters.
        """
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42
        }
    
    def _initialize_model(self):
        """Initialize the Gradient Boosting model."""
        self.model = GradientBoostingClassifier(**self.params)
    
    def get_param_grid(self):
        """
        Get parameter grid for hyperparameter tuning.
        
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }


class ModelFactory:
    """
    Factory class for creating different machine learning models.
    
    This class provides a centralized way to create and configure
    machine learning models for the Titanic survival prediction project.
    It implements the factory pattern to instantiate different model types
    with a standardized interface.
    """
    
    # Registry of available models
    _models = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'decision_tree': DecisionTreeModel,
        'svm': SVMModel,
        'knn': KNeighborsModel,
        'gradient_boosting': GradientBoostingModel
    }
    
    @classmethod
    def create_model(cls, model_type, **kwargs):
        """
        Create and return a machine learning model based on the model type.
        
        Args:
            model_type (str): Type of model to create (e.g., 'random_forest').
            **kwargs: Additional parameters to pass to the model constructor.
            
        Returns:
            TitanicModel: An instance of the requested machine learning model.
            
        Raises:
            ValueError: If the requested model is not supported.
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Model '{model_type}' not supported. "
                f"Available models: {cls.get_available_models()}"
            )
        
        logger.info(f"Creating model of type '{model_type}'")
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls):
        """
        Get a list of available model types.
        
        Returns:
            list: List of available model type strings.
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_default_params(cls, model_type):
        """
        Get the default parameters for a specific model type.
        
        Args:
            model_type (str): Type of model to get parameters for.
            
        Returns:
            dict: Default parameters for the specified model.
            
        Raises:
            ValueError: If the requested model is not supported.
        """
        if model_type not in cls._models:
            raise ValueError(
                f"Model '{model_type}' not supported. "
                f"Available models: {cls.get_available_models()}"
            )
        
        # Create a temporary instance to get default params
        model_class = cls._models[model_type]
        temp_instance = model_class()
        return temp_instance.get_default_params()
    
    @classmethod
    def register_model(cls, model_type, model_class):
        """
        Register a new model type in the factory.
        
        Args:
            model_type (str): Name of the model type to register.
            model_class (class): Class that implements the model.
            
        Raises:
            TypeError: If model_class doesn't inherit from TitanicModel.
        """
        if not issubclass(model_class, TitanicModel):
            raise TypeError(
                f"Model class must inherit from TitanicModel. "
                f"Got {model_class.__name__} instead."
            )
        
        cls._models[model_type] = model_class
        logger.info(f"Registered new model type: '{model_type}'")
