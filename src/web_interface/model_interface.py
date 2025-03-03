# -*- coding: utf-8 -*-
"""
Model Interface Module

This module provides an interface between the web application and the model components.
It handles data loading, model training, and predictions for the Titanic Survival Predictor.
"""

import logging
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer
from src.modelling.model_evaluator import ModelEvaluator
from src.feature_engineering.feature_creator import FeatureCreator
from src.feature_engineering.feature_selector import FeatureSelector
from src.utilities.submission_generator import SubmissionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


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
        logger.info("Initializing ModelInterface")
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_creator = FeatureCreator()
        self.feature_selector = FeatureSelector()
        self.model_factory = ModelFactory()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.submission_generator = SubmissionGenerator()
        
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.trained_models = {}
        self.training_history = {}
        self.model_evaluations = {}
    
    def load_data(self):
        """
        Load and preprocess training and testing data.
        
        Returns:
            bool: True if data loading was successful, False otherwise.
        """
        logger.info("Loading data")
        try:
            # Load data
            self.train_data = self.data_loader.load_train_data()
            self.test_data = self.data_loader.load_test_data()
            
            if self.train_data is None or self.test_data is None:
                logger.error("Failed to load data")
                return False
            
            logger.info(f"Data loaded successfully. Train shape: {self.train_data.shape}, Test shape: {self.test_data.shape}")
            return True
        except Exception as e:
            logger.error(f"Error in load_data: {e}")
            return False
    
    def prepare_data_for_training(self, feature_engineering=False):
        """
        Prepare data for model training, optionally with feature engineering.
        
        Args:
            feature_engineering (bool, optional): Whether to apply feature engineering.
            
        Returns:
            bool: True if data preparation was successful, False otherwise.
        """
        logger.info(f"Preparing data for training with feature_engineering={feature_engineering}")
        
        try:
            if self.train_data is None:
                success = self.load_data()
                if not success:
                    logger.error("Failed to load data during prepare_data_for_training")
                    return False
            
            # Define target and features
            self.y_train = self.train_data['Survived']
            
            # Basic preprocessing
            logger.info("Applying basic preprocessing")
            train_processed = self.preprocessor.fit_transform(self.train_data)
            test_processed = self.preprocessor.transform(self.test_data)
            
            # Feature engineering if requested
            if feature_engineering:
                logger.info("Applying feature engineering")
                train_processed = self.feature_creator.create_all_features(train_processed)
                test_processed = self.feature_creator.create_all_features(test_processed)
            
            # Select features
            logger.info("Selecting features")
            features_to_exclude = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
            self.X_train = train_processed.drop([col for col in features_to_exclude if col in train_processed.columns], axis=1)
            self.X_test = test_processed.drop([col for col in features_to_exclude if col in test_processed.columns], axis=1)
            
            # Ensure X_train and X_test have the same columns
            for col in self.X_train.columns:
                if col not in self.X_test.columns:
                    self.X_test[col] = 0
                    logger.warning(f"Adding missing column {col} to X_test with zeros")
            
            for col in self.X_test.columns:
                if col not in self.X_train.columns:
                    logger.warning(f"Removing column {col} from X_test as it's not in X_train")
            
            self.X_test = self.X_test[self.X_train.columns]
            
            logger.info(f"Data prepared successfully. X_train shape: {self.X_train.shape}, X_test shape: {self.X_test.shape}")
            return True
        except Exception as e:
            logger.error(f"Error in prepare_data_for_training: {e}")
            return False
    
    def train_model(self, model_name, hyperparameter_tuning=False, feature_engineering=False, **kwargs):
        """
        Train a model with the specified name and parameters.
        
        Args:
            model_name (str): Name of the model to train.
            hyperparameter_tuning (bool, optional): Whether to perform hyperparameter tuning.
            feature_engineering (bool, optional): Whether to apply feature engineering.
            **kwargs: Additional parameters to pass to the model constructor.
            
        Returns:
            bool: True if model training was successful, False otherwise.
        """
        logger.info(f"Training model {model_name} with hyperparameter_tuning={hyperparameter_tuning}, feature_engineering={feature_engineering}")
        
        try:
            # Prepare data for training
            if self.X_train is None or self.y_train is None or feature_engineering:
                success = self.prepare_data_for_training(feature_engineering=feature_engineering)
                if not success:
                    logger.error("Failed to prepare data for training")
                    return False
            
            # Create and train the model
            start_time = datetime.now()
            
            # Get model
            model = self.model_factory.get_model(model_name, **kwargs)
            
            # Apply hyperparameter tuning if requested
            if hyperparameter_tuning:
                logger.info(f"Performing hyperparameter tuning for {model_name}")
                param_grid = self._get_param_grid(model_name)
                model = self.model_trainer.tune_hyperparameters(model, self.X_train, self.y_train, param_grid)
            
            # Train the model
            trained_model = self.model_trainer.train(model, self.X_train, self.y_train)
            
            # Evaluate the model
            cv_results = self.model_trainer.cross_validate(model, self.X_train, self.y_train)
            metrics = self.model_evaluator.evaluate_model(trained_model, self.X_train, self.y_train)
            
            # Calculate training time
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Get predictions for ROC curve
            y_pred = trained_model.predict(self.X_train)
            y_pred_proba = None
            if hasattr(trained_model, 'predict_proba'):
                y_pred_proba = trained_model.predict_proba(self.X_train)[:, 1]
            
            # Store the model and metrics
            self.trained_models[model_name] = trained_model
            self.model_evaluations[model_name] = {
                'metrics': metrics,
                'cv_results': cv_results,
                'training_time': training_time,
                'params': model.get_params(),
                'feature_importance': self._get_feature_importance(trained_model, model_name),
                'y_true': self.y_train.values,  # Store for ROC curves
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Save the model and its evaluations
            self._save_model(model_name, trained_model)
            self._save_model_evaluation(model_name, self.model_evaluations[model_name])
            
            logger.info(f"Successfully trained {model_name}. Metrics: {metrics}")
            return True
        except Exception as e:
            logger.error(f"Error in train_model for {model_name}: {e}")
            return False
    
    def predict_survival(self, passenger_data, model_name='random_forest'):
        """
        Predict survival for a single passenger.
        
        Args:
            passenger_data (dict): Dictionary containing passenger information.
            model_name (str, optional): Name of the model to use for prediction.
            
        Returns:
            dict: Prediction results including survival probability and explanations.
        """
        logger.info(f"Predicting survival for a passenger using {model_name}")
        
        try:
            # Create DataFrame from passenger data
            passenger_df = pd.DataFrame([passenger_data])
            
            # Load model if not already loaded
            model = self._get_model(model_name)
            if model is None:
                logger.error(f"Model {model_name} not found")
                return None
            
            # Prepare data
            if self.preprocessor is None:
                logger.error("Preprocessor not initialized")
                return None
            
            # Preprocess the passenger data
            processed_data = self.preprocessor.transform(passenger_df)
            
            # Apply feature engineering if needed
            if hasattr(model, 'feature_engineering') and model.feature_engineering:
                processed_data = self.feature_creator.create_all_features(processed_data)
            
            # Prepare features for prediction
            features_to_exclude = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin']
            X = processed_data.drop([col for col in features_to_exclude if col in processed_data.columns], axis=1)
            
            # Ensure all required columns are present
            for col in self.X_train.columns:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.X_train.columns]
            
            # Make prediction
            survival_prob = model.predict_proba(X)[0, 1]
            survival = 1 if survival_prob >= 0.5 else 0
            
            # Get feature importance for explanation
            feature_importance = self._get_feature_importance(model, model_name)
            
            # Create key factors for explanation
            key_factors = []
            if feature_importance is not None:
                # Get top factors
                for feature, importance in feature_importance[:4]:
                    impact = importance * 100
                    key_factors.append({
                        'factor': feature,
                        'impact': impact
                    })
            
            # Get similar passengers from training data
            similar_passengers = self._get_similar_passengers(passenger_data, survival)
            
            # Create result
            result = {
                'survived': survival,
                'probability': survival_prob,
                'key_factors': key_factors,
                'similar_passengers': similar_passengers
            }
            
            logger.info(f"Prediction result: survival probability {survival_prob:.2f}")
            return result
        except Exception as e:
            logger.error(f"Error in predict_survival: {e}")
            return None
    
    def generate_kaggle_submission(self, model_name, description=None, file_name=None):
        """
        Generate a Kaggle submission file for the specified model.
        
        Args:
            model_name (str): Name of the model to use for submission.
            description (str, optional): Description of the submission for tracking.
            file_name (str, optional): Custom file name for the submission.
            
        Returns:
            dict: Submission details including path and validation results.
        """
        logger.info(f"Generating Kaggle submission for model {model_name}")
        
        try:
            # Ensure data is prepared
            if self.X_test is None:
                success = self.prepare_data_for_training()
                if not success:
                    logger.error("Failed to prepare data for submission")
                    return {
                        'success': False,
                        'error': "Failed to prepare data for submission"
                    }
            
            # Get the model
            model = self._get_model(model_name)
            if model is None:
                logger.error(f"Model {model_name} not found")
                return {
                    'success': False,
                    'error': f"Model {model_name} not found or could not be loaded"
                }
            
            # Generate submission
            submission_path = self.submission_generator.generate_submission(
                model, 
                self.X_test, 
                file_name=file_name,
                description=description
            )
            
            # Validate submission
            validation_results = self.submission_generator.validate_submission(submission_path)
            
            submission_details = {
                'success': True,
                'path': submission_path,
                'file_name': os.path.basename(submission_path),
                'model_name': model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'validation': validation_results
            }
            
            logger.info(f"Submission generated successfully: {submission_path}")
            return submission_details
        except Exception as e:
            logger.error(f"Error generating Kaggle submission: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_submissions(self):
        """
        List all available submission files.
        
        Returns:
            pandas.DataFrame: List of submission files with metadata.
        """
        try:
            submissions_df = self.submission_generator.list_submissions()
            return submissions_df
        except Exception as e:
            logger.error(f"Error listing submissions: {e}")
            return pd.DataFrame()
    
    def compare_model_submissions(self, model_names):
        """
        Generate and compare submissions from different models.
        
        Args:
            model_names (list): List of model names to compare.
            
        Returns:
            dict: Comparison results.
        """
        logger.info(f"Comparing submissions for models: {model_names}")
        
        try:
            # Ensure data is prepared
            if self.X_test is None:
                success = self.prepare_data_for_training()
                if not success:
                    logger.error("Failed to prepare data for submission comparison")
                    return {
                        'success': False,
                        'error': "Failed to prepare data for submission comparison"
                    }
            
            # Generate submissions for each model
            submission_paths = []
            for model_name in model_names:
                model = self._get_model(model_name)
                if model is None:
                    logger.warning(f"Model {model_name} not found, skipping")
                    continue
                
                # Generate submission
                file_name = f"comparison_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                submission_path = self.submission_generator.generate_submission(
                    model, 
                    self.X_test, 
                    file_name=file_name,
                    description=f"Generated for model comparison: {model_name}"
                )
                submission_paths.append(submission_path)
            
            if not submission_paths:
                return {
                    'success': False,
                    'error': "No models were found for comparison"
                }
            
            # Compare submissions
            comparison_results = self.submission_generator.compare_submissions(
                submission_paths=submission_paths, 
                names=model_names
            )
            
            return {
                'success': True,
                'comparison': comparison_results,
                'submissions': submission_paths
            }
        except Exception as e:
            logger.error(f"Error comparing model submissions: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_performance(self, model_name=None):
        """
        Get performance metrics for a specific model or all models.
        
        Args:
            model_name (str, optional): Name of the model. If None, returns metrics for all models.
            
        Returns:
            dict or list: Performance metrics for the specified model or all models.
        """
        try:
            if model_name:
                # First check if we have the model evaluation in memory
                if model_name in self.model_evaluations:
                    logger.info(f"Found model evaluation for {model_name} in memory")
                    return self.model_evaluations[model_name]
                else:
                    # Try to load from disk
                    logger.info(f"Attempting to load model evaluation for {model_name} from disk")
                    evaluation = self._load_model_evaluation(model_name)
                    if evaluation:
                        # Store in memory for future use
                        self.model_evaluations[model_name] = evaluation
                        return evaluation
                    else:
                        logger.warning(f"Model {model_name} evaluation not found")
                        return None
            else:
                # Return all model evaluations
                all_models = []
                
                # First check in-memory evaluations
                for name, evaluation in self.model_evaluations.items():
                    model_info = {
                        'name': name,
                        'metrics': evaluation['metrics'],
                        'training_time': evaluation['training_time']
                    }
                    all_models.append(model_info)
                
                # Then check saved evaluations
                saved_models = self._list_saved_model_evaluations()
                for name in saved_models:
                    if name not in [model['name'] for model in all_models]:
                        evaluation = self._load_model_evaluation(name)
                        if evaluation:
                            model_info = {
                                'name': name,
                                'metrics': evaluation['metrics'],
                                'training_time': evaluation.get('training_time', 0)
                            }
                            all_models.append(model_info)
                
                return all_models
        except Exception as e:
            logger.error(f"Error in get_model_performance: {e}")
            return None
    
    def get_feature_importance(self, model_name='random_forest'):
        """
        Get feature importance for a specified model.
        
        Args:
            model_name (str, optional): Name of the model.
            
        Returns:
            list: List of (feature, importance) tuples.
        """
        try:
            # First check if we have the model evaluation in memory
            if model_name in self.model_evaluations and 'feature_importance' in self.model_evaluations[model_name]:
                return self.model_evaluations[model_name]['feature_importance']
            else:
                # Try to load from disk
                evaluation = self._load_model_evaluation(model_name)
                if evaluation and 'feature_importance' in evaluation:
                    return evaluation['feature_importance']
                
                # If no saved evaluation or feature importance not in evaluation,
                # try to load model and compute feature importance
                model = self._get_model(model_name)
                if model:
                    return self._get_feature_importance(model, model_name)
                else:
                    logger.warning(f"Model {model_name} not found")
                    return None
        except Exception as e:
            logger.error(f"Error in get_feature_importance: {e}")
            return None
    
    def list_available_models(self):
        """
        List all available trained models.
        
        Returns:
            list: List of model names and their metrics.
        """
        try:
            models = []
            # Check trained models
            for name in self.trained_models.keys():
                metrics = self.model_evaluations.get(name, {}).get('metrics', {})
                models.append({
                    'name': name,
                    'accuracy': metrics.get('accuracy', 0),
                    'trained': True
                })
            
            # Check saved models
            for name in self._list_saved_models():
                if name not in [model['name'] for model in models]:
                    # Try to load evaluation to get accuracy
                    evaluation = self._load_model_evaluation(name)
                    accuracy = 0
                    if evaluation and 'metrics' in evaluation:
                        accuracy = evaluation['metrics'].get('accuracy', 0)
                    
                    models.append({
                        'name': name,
                        'accuracy': accuracy,
                        'trained': False
                    })
            
            return models
        except Exception as e:
            logger.error(f"Error in list_available_models: {e}")
            return []
    
    def _get_param_grid(self, model_name):
        """
        Get parameter grid for hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model.
            
        Returns:
            dict: Parameter grid for GridSearchCV.
        """
        if model_name == 'logistic_regression':
            return {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'decision_tree':
            return {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        elif model_name == 'svm':
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        elif model_name == 'knn':
            return {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
        else:
            logger.warning(f"No parameter grid defined for {model_name}, using empty grid")
            return {}
    
    def _get_feature_importance(self, model, model_name):
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model.
            model_name (str): Name of the model.
            
        Returns:
            list: List of (feature, importance) tuples.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                features = self.X_train.columns
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = abs(model.coef_[0])
                features = self.X_train.columns
            else:
                # Use feature selector for models without built-in importance
                importance_df = self.feature_selector.get_feature_importance(
                    self.X_train, self.y_train, method='random_forest'
                )
                importances = importance_df['Importance'].values
                features = importance_df['Feature'].values
            
            # Create list of (feature, importance) tuples
            feature_importance = [(feature, importance) for feature, importance in zip(features, importances)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return feature_importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None
    
    def _get_similar_passengers(self, passenger_data, survival):
        """
        Find similar passengers in the training data.
        
        Args:
            passenger_data (dict): Passenger data.
            survival (int): Predicted survival (0 or 1).
            
        Returns:
            list: List of similar passengers.
        """
        try:
            # Get relevant filters
            pclass = passenger_data.get('Pclass')
            sex = passenger_data.get('Sex')
            age_range = (passenger_data.get('Age', 30) - 10, passenger_data.get('Age', 30) + 10)
            
            # Filter training data
            filtered_data = self.train_data[
                (self.train_data['Pclass'] == pclass) &
                (self.train_data['Sex'] == sex) &
                (self.train_data['Survived'] == survival)
            ]
            
            if 'Age' in passenger_data and passenger_data['Age'] is not None:
                filtered_data = filtered_data[
                    (filtered_data['Age'] >= age_range[0]) &
                    (filtered_data['Age'] <= age_range[1])
                ]
            
            # Get top 3 similar passengers
            similar_passengers = []
            for _, row in filtered_data.head(3).iterrows():
                passenger = {
                    'name': row['Name'],
                    'age': row['Age'],
                    'sex': row['Sex'],
                    'pclass': f"{row['Pclass']}st" if row['Pclass'] == 1 else 
                              f"{row['Pclass']}nd" if row['Pclass'] == 2 else 
                              f"{row['Pclass']}rd",
                    'fare': row['Fare'],
                    'embarked': row['Embarked'],
                    'survived': row['Survived']
                }
                similar_passengers.append(passenger)
            
            return similar_passengers
        except Exception as e:
            logger.error(f"Error getting similar passengers: {e}")
            return []
    
    def _save_model(self, model_name, model):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model.
            model: Trained model to save.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model {model_name} saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def _save_model_evaluation(self, model_name, evaluation):
        """
        Save model evaluation results to disk.
        
        Args:
            model_name (str): Name of the model.
            evaluation (dict): Model evaluation metrics and data.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create a copy to avoid modifying the original
            eval_copy = evaluation.copy()
            
            # Convert numpy arrays to lists for pickling
            for key in ['y_true', 'y_pred', 'y_pred_proba']:
                if key in eval_copy and eval_copy[key] is not None:
                    if hasattr(eval_copy[key], 'tolist'):
                        eval_copy[key] = eval_copy[key].tolist()
            
            # Save to disk
            eval_path = os.path.join(MODEL_DIR, f"{model_name}_eval.pkl")
            with open(eval_path, 'wb') as f:
                pickle.dump(eval_copy, f)
            logger.info(f"Model evaluation for {model_name} saved to {eval_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving evaluation for {model_name}: {e}")
            return False
    
    def _get_model(self, model_name):
        """
        Get a model, loading from disk if necessary.
        
        Args:
            model_name (str): Name of the model.
            
        Returns:
            object: The model, or None if not found.
        """
        try:
            # Check if model is already in memory
            if model_name in self.trained_models:
                return self.trained_models[model_name]
            
            # Try to load from disk
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.trained_models[model_name] = model
                logger.info(f"Loaded model {model_name} from {model_path}")
                return model
            else:
                logger.warning(f"Model {model_name} not found on disk")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_model_evaluation(self, model_name):
        """
        Load model evaluation from disk.
        
        Args:
            model_name (str): Name of the model.
            
        Returns:
            dict: Model evaluation data, or None if not found.
        """
        try:
            eval_path = os.path.join(MODEL_DIR, f"{model_name}_eval.pkl")
            if os.path.exists(eval_path):
                with open(eval_path, 'rb') as f:
                    evaluation = pickle.load(f)
                logger.info(f"Loaded evaluation for {model_name} from {eval_path}")
                return evaluation
            else:
                logger.warning(f"Evaluation for {model_name} not found on disk")
                return None
        except Exception as e:
            logger.error(f"Error loading evaluation for {model_name}: {e}")
            return None
    
    def _list_saved_models(self):
        """
        List all saved models on disk.
        
        Returns:
            list: List of model names.
        """
        try:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and not f.endswith('_eval.pkl')]
            model_names = [os.path.splitext(f)[0] for f in model_files]
            return model_names
        except Exception as e:
            logger.error(f"Error listing saved models: {e}")
            return []
    
    def _list_saved_model_evaluations(self):
        """
        List all saved model evaluations on disk.
        
        Returns:
            list: List of model names with evaluations.
        """
        try:
            eval_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_eval.pkl')]
            model_names = [os.path.splitext(f)[0].replace('_eval', '') for f in eval_files]
            return model_names
        except Exception as e:
            logger.error(f"Error listing saved model evaluations: {e}")
            return []
