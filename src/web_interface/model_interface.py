# -*- coding: utf-8 -*-
"""
Model Interface Module

This module provides an interface between the web application and the model components.
"""

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer
from src.modelling.model_evaluator import ModelEvaluator
from src.feature_engineering.feature_creator import FeatureCreator
from src.feature_engineering.feature_selector import FeatureSelector


class ModelInterface:
    """
    ModelInterface class for interfacing between web app and model components.
    
    This class provides methods to load data, train models, and make predictions
    that can be called from the web interface.
    """
    
    def __init__(self):
        """
        Initialize the ModelInterface with necessary components.
        """
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_creator = FeatureCreator()
        self.feature_selector = FeatureSelector()
        self.model_factory = ModelFactory()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.trained_models = {}
    
    def load_data(self):
        """
        Load and preprocess training and testing data.
        
        Returns:
            bool: True if data loading was successful, False otherwise.
        """
        # Load data
        self.train_data = self.data_loader.load_train_data()
        self.test_data = self.data_loader.load_test_data()
        
        if self.train_data is None or self.test_data is None:
            return False
        
        return True
    
    def prepare_data_for_training(self, feature_engineering=False):
        """
        Prepare data for model training, optionally with feature engineering.
        
        Args:
            feature_engineering (bool, optional): Whether to apply feature engineering.
            
        Returns:
            bool: True if data preparation was successful, False otherwise.
        """
        if self.train_data is None:
            success = self.load_data()
            if not success:
                return False
        
        # Define target and features
        self.y_train = self.train_data['Survived']
        
        # Basic preprocessing
        train_processed = self.preprocessor.fit_transform(self.train_data)
        test_processed = self.preprocessor.transform(self.test_data)
        
        # Feature engineering if requested
        if feature_engineering:
            train_processed = self.feature_creator.create_family_size_feature(train_processed)
            test_processed = self.feature_creator.create_family_size_feature(test_processed)
            
            # Add more feature engineering as needed
        
        # Select features
        features_to_exclude = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
        self.X_train = train_processed.drop([col for col in features_to_exclude if col in train_processed.columns], axis=1)
        self.X_test = test_processed.drop([col for col in features_to_exclude if col in test_processed.columns], axis=1)
        
        return True
    
    def train_model(self, model_name, **kwargs):
        """
        Train a model with the specified name and parameters.
        
        Args:
            model_name (str): Name of the model to train.
            **kwargs: Additional parameters to pass to the model constructor.
            
        Returns:
            bool: True if model training was successful, False otherwise.
        """
        if self.X_train is None or self.y_train is None:
            success = self.prepare_data_for_training()
            if not success:
                return False
        
        # Create and train the model
        try:
            model = self.model_factory.get_model(model_name, **kwargs)
            self.model_trainer.train_model(model, self.X_train, self.y_train, model_name)
            self.trained_models[model_name] = model
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def evaluate_models(self):
        """
        Evaluate all trained models and return performance metrics.
        
        Returns:
            pandas.DataFrame: DataFrame with model evaluation metrics.
        """
        if not self.trained_models:
            print("No trained models available for evaluation.")
            return None
        
        if self.X_train is None or self.y_train is None:
            success = self.prepare_data_for_training()
            if not success:
                return None
        
        # Evaluate models
        return self.model_evaluator.compare_models(self.trained_models, self.X_train, self.y_train)
