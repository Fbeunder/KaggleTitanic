# -*- coding: utf-8 -*-
"""
Model Trainer Module

This module is responsible for training machine learning models on the Titanic dataset,
including features for cross-validation, hyperparameter tuning, and data splitting.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from src.modelling.model_factory import ModelFactory, TitanicModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    ModelTrainer class for training machine learning models.
    
    This class handles the training process for machine learning models
    on the Titanic dataset, including cross-validation, hyperparameter tuning,
    and proper data splitting for training, validation, and testing.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state (int, optional): Random state for reproducibility.
        """
        self.random_state = random_state
        self.trained_models = {}
        self.model_performance = {}
    
    def train(self, model, X_train, y_train, model_name=None):
        """
        Train a model on the provided data.
        
        Args:
            model: Machine learning model to train (instance of TitanicModel).
            X_train (numpy.ndarray or pandas.DataFrame): Feature matrix for training.
            y_train (numpy.ndarray or pandas.Series): Target vector for training.
            model_name (str, optional): Name to assign to the trained model.
                                       If None, uses the model's name attribute.
                                       
        Returns:
            TitanicModel: The trained model.
        """
        logger.info(f"Training model: {model.name if hasattr(model, 'name') else type(model).__name__}")
        
        # Train the model
        if isinstance(model, TitanicModel):
            model.fit(X_train, y_train)
        else:
            logger.warning("Model is not an instance of TitanicModel. Training using fit method directly.")
            model.fit(X_train, y_train)
        
        # Determine model name if not provided
        if model_name is None:
            model_name = model.name if hasattr(model, 'name') else type(model).__name__
        
        # Store the trained model
        self.trained_models[model_name] = model
        logger.info(f"Model '{model_name}' trained successfully")
        
        return model
    
    def cross_validate(self, model, X, y, cv=5, scoring='accuracy', return_train_score=False):
        """
        Perform cross-validation on a model.
        
        Args:
            model: Machine learning model to evaluate.
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y (numpy.ndarray or pandas.Series): Target vector.
            cv (int or cross-validator, optional): Cross-validation strategy.
            scoring (str or callable, optional): Scoring metric to use.
            return_train_score (bool, optional): Whether to return training scores.
            
        Returns:
            dict: Dictionary containing cross-validation results, including:
                - mean_test_score: Mean score on test folds
                - std_test_score: Standard deviation of test scores
                - cv_results: Array of scores from each fold
                - mean_train_score: Mean score on training folds (if return_train_score=True)
                - std_train_score: Standard deviation of training scores (if return_train_score=True)
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Ensure proper cross-validation strategy
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        if isinstance(model, TitanicModel):
            model_instance = model.model
        else:
            model_instance = model
            
        scores = cross_val_score(model_instance, X, y, cv=cv, scoring=scoring)
        
        # Calculate metrics
        result = {
            'mean_test_score': np.mean(scores),
            'std_test_score': np.std(scores),
            'cv_results': scores
        }
        
        # Calculate training scores if requested
        if return_train_score:
            train_scores = []
            for train_idx, _ in cv.split(X, y):
                X_train_cv, y_train_cv = X.iloc[train_idx], y.iloc[train_idx]
                model_clone = clone(model_instance)
                model_clone.fit(X_train_cv, y_train_cv)
                train_score = model_clone.score(X_train_cv, y_train_cv)
                train_scores.append(train_score)
                
            result['mean_train_score'] = np.mean(train_scores)
            result['std_train_score'] = np.std(train_scores)
        
        logger.info(f"Cross-validation results: mean={result['mean_test_score']:.4f}, std={result['std_test_score']:.4f}")
        
        return result
    
    def train_test_split_with_validation(self, data, target_col, test_size=0.2, val_size=0.2, stratify=True):
        """
        Split data into training, validation, and test sets.
        
        Args:
            data (pandas.DataFrame): Dataset to split.
            target_col (str): Name of the target column.
            test_size (float, optional): Proportion of data to use for testing.
            val_size (float, optional): Proportion of training data to use for validation.
            stratify (bool, optional): Whether to use stratified splitting.
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test) split datasets.
        """
        logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
        
        # Extract features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Stratification
        stratify_data = y if stratify else None
        
        # First split: training+validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_data
        )
        
        # Second split: training vs validation
        # Recalculate stratify with the new subset
        stratify_train_val = y_train_val if stratify else None
        
        # Calculate adjusted validation size relative to the train_val set
        adjusted_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=adjusted_val_size, 
            random_state=self.random_state,
            stratify=stratify_train_val
        )
        
        logger.info(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_best_hyperparameters(self, model_type, param_grid=None, X=None, y=None, cv=5, scoring='accuracy'):
        """
        Find the best hyperparameters for a specific model type.
        
        Args:
            model_type (str): Type of model to tune (e.g., 'random_forest').
            param_grid (dict, optional): Grid of parameters to search.
                                       If None, uses the model's default grid.
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y (numpy.ndarray or pandas.Series): Target vector.
            cv (int, optional): Number of cross-validation folds.
            scoring (str or callable, optional): Scoring metric to optimize.
            
        Returns:
            dict: Dictionary containing:
                - best_params: Best hyperparameters found
                - best_score: Best score achieved
                - cv_results: Full results from grid search
        """
        if X is None or y is None:
            raise ValueError("X and y must be provided for hyperparameter tuning")
            
        logger.info(f"Finding best hyperparameters for model type: {model_type}")
        
        # Create model instance
        model = ModelFactory.create_model(model_type)
        
        # Get parameter grid if not provided
        if param_grid is None:
            param_grid = model.get_param_grid()
            
        if not param_grid:
            logger.warning(f"No parameter grid defined for {model_type}. Skipping hyperparameter tuning.")
            return {
                'best_params': {},
                'best_score': None,
                'cv_results': None
            }
        
        # Create grid search
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Perform grid search
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_multiple_models(self, models, X_train, y_train, X_val=None, y_val=None):
        """
        Train multiple models and evaluate them on validation data if provided.
        
        Args:
            models (dict or list): Dictionary of {name: model} or list of models to train.
            X_train (numpy.ndarray or pandas.DataFrame): Training feature matrix.
            y_train (numpy.ndarray or pandas.Series): Training target vector.
            X_val (numpy.ndarray or pandas.DataFrame, optional): Validation feature matrix.
            y_val (numpy.ndarray or pandas.Series, optional): Validation target vector.
            
        Returns:
            dict: Dictionary of trained models and their performance if validation data is provided.
        """
        # Convert list to dictionary if necessary
        if isinstance(models, list):
            models_dict = {
                model.name if hasattr(model, 'name') else f"model_{i}": model
                for i, model in enumerate(models)
            }
        else:
            models_dict = models
            
        logger.info(f"Training {len(models_dict)} models")
        
        # Train each model
        for model_name, model in models_dict.items():
            self.train(model, X_train, y_train, model_name=model_name)
            
            # Evaluate on validation data if provided
            if X_val is not None and y_val is not None:
                # Get predictions
                if isinstance(model, TitanicModel):
                    y_pred = model.predict(X_val)
                else:
                    y_pred = model.predict(X_val)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_val, y_pred)
                self.model_performance[model_name] = accuracy
                logger.info(f"Validation accuracy for {model_name}: {accuracy:.4f}")
                
        return {
            'trained_models': self.trained_models,
            'performance': self.model_performance if X_val is not None else None
        }
