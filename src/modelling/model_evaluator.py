# -*- coding: utf-8 -*-
"""
Model Evaluator Module

This module is responsible for evaluating machine learning models on the Titanic dataset,
including performance metrics, visualization, and model comparison.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from src.modelling.model_factory import TitanicModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    ModelEvaluator class for evaluating machine learning models.
    
    This class handles the evaluation of trained machine learning models
    using various performance metrics and visualization techniques.
    """
    
    def __init__(self):
        """
        Initialize the ModelEvaluator.
        """
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name=None):
        """
        Evaluate a model using multiple metrics.
        
        Args:
            model: Trained machine learning model to evaluate.
            X_test (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y_test (numpy.ndarray or pandas.Series): True target values.
            model_name (str, optional): Name for the model in result storage.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model: {model_name if model_name else 'Unnamed model'}")
        
        # Make predictions
        if isinstance(model, TitanicModel):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).shape[1] > 1 else model.predict_proba(X_test)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
            # Some models don't have predict_proba
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except (AttributeError, IndexError):
                logger.warning("Model doesn't support probability prediction")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        # Add AUC if probabilities are available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Store results if model name is provided
        if model_name:
            self.evaluation_results[model_name] = {
                'metrics': metrics,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        logger.info(f"Evaluation metrics: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate various performance metrics for a set of predictions.
        
        Args:
            y_true (numpy.ndarray or pandas.Series): True target values.
            y_pred (numpy.ndarray or pandas.Series): Predicted target values.
            y_pred_proba (numpy.ndarray, optional): Predicted probabilities.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred, normalize=False):
        """
        Generate a confusion matrix for the given predictions.
        
        Args:
            y_true (numpy.ndarray or pandas.Series): True target values.
            y_pred (numpy.ndarray or pandas.Series): Predicted target values.
            normalize (bool or str, optional): Normalization option.
                'true' for normalization by row, 'pred' for by column,
                'all' for by total, and False for no normalization.
            
        Returns:
            numpy.ndarray: Confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        return cm
    
    def plot_confusion_matrix(self, y_true=None, y_pred=None, model_name=None, normalize=False, 
                             title=None, cmap=plt.cm.Blues, figsize=(10, 8), annot=True):
        """
        Plot a confusion matrix.
        
        Args:
            y_true (numpy.ndarray or pandas.Series, optional): True target values.
            y_pred (numpy.ndarray or pandas.Series, optional): Predicted target values.
            model_name (str, optional): Name of a previously evaluated model.
            normalize (bool or str, optional): Normalization option.
            title (str, optional): Title for the plot.
            cmap (matplotlib.colors.Colormap, optional): Color map for the plot.
            figsize (tuple, optional): Figure size.
            annot (bool, optional): Whether to annotate cells.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Get data from stored results if model_name is provided
        if model_name and not (y_true is not None and y_pred is not None):
            if model_name not in self.evaluation_results:
                raise ValueError(f"Model '{model_name}' not found in evaluation results")
            
            y_true = self.evaluation_results[model_name]['y_true']
            y_pred = self.evaluation_results[model_name]['y_pred']
        
        if y_true is None or y_pred is None:
            raise ValueError("Either provide y_true and y_pred, or a valid model_name")
        
        # Generate confusion matrix
        cm = self.get_confusion_matrix(y_true, y_pred, normalize=normalize)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(cm, annot=annot, fmt='.2f' if normalize else 'd', 
                    cmap=cmap, ax=ax, cbar=True)
        
        # Set labels
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            norm_str = f" (normalized: {normalize})" if normalize else ""
            ax.set_title(f"Confusion Matrix{norm_str}")
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true=None, y_pred_proba=None, model_name=None, 
                      title=None, figsize=(10, 8)):
        """
        Plot a ROC curve.
        
        Args:
            y_true (numpy.ndarray or pandas.Series, optional): True target values.
            y_pred_proba (numpy.ndarray, optional): Predicted probabilities.
            model_name (str, optional): Name of a previously evaluated model.
            title (str, optional): Title for the plot.
            figsize (tuple, optional): Figure size.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Get data from stored results if model_name is provided
        if model_name and not (y_true is not None and y_pred_proba is not None):
            if model_name not in self.evaluation_results:
                raise ValueError(f"Model '{model_name}' not found in evaluation results")
            
            y_true = self.evaluation_results[model_name]['y_true']
            y_pred_proba = self.evaluation_results[model_name]['y_pred_proba']
            
            if y_pred_proba is None:
                raise ValueError(f"Model '{model_name}' doesn't have probability predictions")
        
        if y_true is None or y_pred_proba is None:
            raise ValueError("Either provide y_true and y_pred_proba, or a valid model_name")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Set labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
        if title:
            ax.set_title(title)
        else:
            model_str = f" for {model_name}" if model_name else ""
            ax.set_title(f"ROC Curve{model_str}")
        
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    
    def compare_models(self, models_dict=None, X=None, y=None, metric='accuracy'):
        """
        Compare multiple models using the same evaluation metric.
        
        Args:
            models_dict (dict, optional): Dictionary of trained models {model_name: model}.
                                       If None, uses previously evaluated models.
            X (numpy.ndarray or pandas.DataFrame, optional): Feature matrix.
            y (numpy.ndarray or pandas.Series, optional): True target values.
            metric (str, optional): Metric to compare models by.
            
        Returns:
            pandas.DataFrame: Comparison of models across evaluation metrics.
        """
        results = {}
        
        # Use previously evaluated models if no new ones are provided
        if models_dict is None:
            if not self.evaluation_results:
                raise ValueError("No evaluated models found and no new models provided")
            
            # Extract metrics from stored evaluation results
            for model_name, result in self.evaluation_results.items():
                results[model_name] = result['metrics']
        else:
            # Evaluate new models
            if X is None or y is None:
                raise ValueError("X and y must be provided when evaluating new models")
            
            for model_name, model in models_dict.items():
                metrics = self.evaluate_model(model, X, y, model_name=model_name)
                results[model_name] = metrics
        
        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame()
        
        for model_name, metrics in results.items():
            # Extract scalar metrics (exclude matrices and reports)
            scalar_metrics = {k: v for k, v in metrics.items() 
                             if not isinstance(v, (np.ndarray, dict))}
            results_df = pd.concat([results_df, pd.DataFrame([scalar_metrics], index=[model_name])])
        
        # Sort by specified metric
        if metric in results_df.columns:
            results_df = results_df.sort_values(by=metric, ascending=False)
        
        return results_df
    
    def plot_model_comparison(self, models_dict=None, X=None, y=None, 
                             metrics=None, figsize=(12, 8)):
        """
        Plot a comparison of multiple models across different metrics.
        
        Args:
            models_dict (dict, optional): Dictionary of trained models {model_name: model}.
                                       If None, uses previously evaluated models.
            X (numpy.ndarray or pandas.DataFrame, optional): Feature matrix.
            y (numpy.ndarray or pandas.Series, optional): True target values.
            metrics (list, optional): List of metrics to include in comparison.
                                   If None, uses ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].
            figsize (tuple, optional): Figure size.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Get comparison DataFrame
        comparison_df = self.compare_models(models_dict, X, y)
        
        # Default metrics to compare
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            # Add roc_auc if available for all models
            if 'roc_auc' in comparison_df.columns and not comparison_df['roc_auc'].isna().any():
                metrics.append('roc_auc')
        
        # Filter metrics that are available
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            raise ValueError("None of the specified metrics are available in the comparison")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        comparison_df[available_metrics].plot(kind='bar', ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.legend(title='Metric')
        
        plt.tight_layout()
        return fig
