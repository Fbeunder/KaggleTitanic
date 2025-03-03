# -*- coding: utf-8 -*-
"""
Feature Selector Module

This module is responsible for selecting the most important features for model training.
It implements various feature selection techniques to identify the most predictive features.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    FeatureSelector class for selecting the most important features.
    
    This class implements various feature selection techniques to identify
    the most predictive features for the Titanic survival prediction.
    """
    
    def __init__(self):
        """
        Initialize the FeatureSelector.
        """
        logger.info("Initializing FeatureSelector")
        self.selected_features = None
        self.feature_importance = None
    
    def select_k_best(self, X, y, k=10, score_func=f_classif):
        """
        Select k best features using statistical tests between features and target.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            k (int, optional): Number of top features to select.
            score_func (callable, optional): Function to score features. Default is f_classif (ANOVA).
                Other options include mutual_info_classif for non-linear relationships.
            
        Returns:
            pandas.DataFrame: DataFrame with only the selected features.
        """
        logger.info(f"Selecting top {k} features using {score_func.__name__}")
        
        # Validate input
        if k > X.shape[1]:
            logger.warning(f"k ({k}) is greater than the number of features ({X.shape[1]}). Setting k to {X.shape[1]}.")
            k = X.shape[1]
        
        # Initialize selector
        selector = SelectKBest(score_func=score_func, k=k)
        
        try:
            # Fit and transform
            X_new = selector.fit_transform(X, y)
            
            # Get selected feature names if input is DataFrame
            if isinstance(X, pd.DataFrame):
                selected_indices = selector.get_support(indices=True)
                selected_features = X.columns[selected_indices]
                self.selected_features = selected_features
                
                # Store feature importance scores
                feature_scores = pd.DataFrame({
                    'Feature': X.columns,
                    'Score': selector.scores_,
                    'P-value': selector.pvalues_ if hasattr(selector, 'pvalues_') else np.nan
                })
                self.feature_importance = feature_scores.sort_values('Score', ascending=False)
                
                # Log selected features
                logger.info(f"Selected features: {', '.join(selected_features)}")
                
                return X[selected_features]
            else:
                self.selected_features = np.arange(X.shape[1])[selector.get_support()]
                logger.info(f"Selected feature indices: {self.selected_features}")
                return X_new
                
        except Exception as e:
            logger.error(f"Error in select_k_best: {e}")
            return X
    
    def select_with_rfe(self, X, y, n_features_to_select=10, step=1, estimator=None):
        """
        Select features using Recursive Feature Elimination with a specified estimator.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            n_features_to_select (int, optional): Number of features to select.
            step (int, optional): Number of features to remove at each iteration.
            estimator (object, optional): Estimator to use for feature ranking. Default is RandomForest.
            
        Returns:
            pandas.DataFrame: DataFrame with only the selected features.
        """
        logger.info(f"Selecting {n_features_to_select} features using RFE")
        
        # Validate input
        if n_features_to_select > X.shape[1]:
            logger.warning(f"n_features_to_select ({n_features_to_select}) is greater than the number of features ({X.shape[1]}). Setting to {X.shape[1]}.")
            n_features_to_select = X.shape[1]
        
        # Default estimator if none provided
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            logger.info("Using default RandomForestClassifier for RFE")
        
        try:
            # Initialize selector
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step,
                verbose=0
            )
            
            # Fit selector
            selector.fit(X, y)
            
            # Get selected feature names if input is DataFrame
            if isinstance(X, pd.DataFrame):
                selected_features = X.columns[selector.support_]
                self.selected_features = selected_features
                
                # Store feature rankings
                feature_ranks = pd.DataFrame({
                    'Feature': X.columns,
                    'Rank': selector.ranking_,
                    'Selected': selector.support_
                })
                self.feature_importance = feature_ranks.sort_values('Rank')
                
                # Log selected features
                logger.info(f"Selected features: {', '.join(selected_features)}")
                
                return X[selected_features]
            else:
                self.selected_features = np.arange(X.shape[1])[selector.support_]
                logger.info(f"Selected feature indices: {self.selected_features}")
                return selector.transform(X)
                
        except Exception as e:
            logger.error(f"Error in select_with_rfe: {e}")
            return X
    
    def select_with_rfecv(self, X, y, cv=5, scoring='accuracy', min_features_to_select=5, estimator=None):
        """
        Select features using RFE with cross-validation to automatically determine the optimal number of features.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            cv (int, optional): Number of cross-validation folds.
            scoring (str, optional): Scoring metric for evaluation.
            min_features_to_select (int, optional): Minimum number of features to select.
            estimator (object, optional): Estimator to use for feature ranking. Default is RandomForest.
            
        Returns:
            pandas.DataFrame: DataFrame with only the selected features.
        """
        logger.info(f"Selecting features using RFECV with {cv}-fold cross-validation")
        
        # Default estimator if none provided
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            logger.info("Using default RandomForestClassifier for RFECV")
        
        try:
            # Initialize selector
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring=scoring,
                min_features_to_select=min_features_to_select,
                verbose=0
            )
            
            # Fit selector
            selector.fit(X, y)
            
            # Get selected feature names if input is DataFrame
            if isinstance(X, pd.DataFrame):
                selected_features = X.columns[selector.support_]
                self.selected_features = selected_features
                
                # Store feature rankings
                feature_ranks = pd.DataFrame({
                    'Feature': X.columns,
                    'Rank': selector.ranking_,
                    'Selected': selector.support_
                })
                self.feature_importance = feature_ranks.sort_values('Rank')
                
                # Log selected features and optimal number
                logger.info(f"Optimal number of features: {selector.n_features_}")
                logger.info(f"Selected features: {', '.join(selected_features)}")
                
                return X[selected_features]
            else:
                self.selected_features = np.arange(X.shape[1])[selector.support_]
                logger.info(f"Selected feature indices: {self.selected_features}")
                logger.info(f"Optimal number of features: {selector.n_features_}")
                return selector.transform(X)
                
        except Exception as e:
            logger.error(f"Error in select_with_rfecv: {e}")
            return X
    
    def get_feature_importance(self, X, y, method='random_forest', **kwargs):
        """
        Get feature importance using a specified method.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            method (str, optional): Method to use for feature importance.
                Options: 'random_forest', 'gradient_boosting', 'permutation', 'lasso'
            **kwargs: Additional parameters for the specific method.
            
        Returns:
            pandas.DataFrame: DataFrame with features and their importance scores.
        """
        logger.info(f"Getting feature importance using {method} method")
        
        if not isinstance(X, pd.DataFrame):
            logger.warning("X is not a pandas DataFrame. Converting to DataFrame with default column names.")
            X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
        
        try:
            importance_df = None
            
            if method == 'random_forest':
                # Extract parameters for RandomForest
                n_estimators = kwargs.get('n_estimators', 100)
                random_state = kwargs.get('random_state', 42)
                
                # Train RandomForest model
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                rf.fit(X, y)
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                })
                
            elif method == 'gradient_boosting':
                # Extract parameters for GradientBoosting
                n_estimators = kwargs.get('n_estimators', 100)
                random_state = kwargs.get('random_state', 42)
                
                # Train GradientBoosting model
                gb = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
                gb.fit(X, y)
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': gb.feature_importances_
                })
                
            elif method == 'permutation':
                # Extract parameters for permutation importance
                n_repeats = kwargs.get('n_repeats', 10)
                random_state = kwargs.get('random_state', 42)
                scoring = kwargs.get('scoring', 'roc_auc')
                
                # Choose base estimator
                estimator_type = kwargs.get('estimator', 'random_forest')
                if estimator_type == 'random_forest':
                    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
                else:
                    estimator = LogisticRegression(random_state=random_state)
                
                # Train model
                estimator.fit(X, y)
                
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    estimator, X, y, n_repeats=n_repeats,
                    random_state=random_state, scoring=scoring
                )
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': perm_importance.importances_mean,
                    'StdDev': perm_importance.importances_std
                })
                
            elif method == 'lasso':
                # Extract parameters for Lasso
                alpha = kwargs.get('alpha', 0.01)
                random_state = kwargs.get('random_state', 42)
                
                # Train Lasso model (for regression tasks)
                # For classification, we'll use LogisticRegression with l1 penalty
                if len(np.unique(y)) > 2 or y.dtype.kind == 'f':  # For multi-class or regression
                    logger.warning("Lasso is using LogisticRegression with l1 penalty for classification tasks.")
                    lasso = LogisticRegression(
                        penalty='l1', solver='liblinear', C=1/alpha,
                        random_state=random_state
                    )
                else:  # For binary classification
                    lasso = LogisticRegression(
                        penalty='l1', solver='liblinear', C=1/alpha,
                        random_state=random_state
                    )
                
                # Train model
                lasso.fit(X, y)
                
                # Get feature importance (absolute value of coefficients)
                if hasattr(lasso, 'coef_'):
                    if lasso.coef_.ndim > 1:  # For multi-class
                        importance = np.mean(np.abs(lasso.coef_), axis=0)
                    else:  # For binary classification or regression
                        importance = np.abs(lasso.coef_)
                        
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importance
                    })
                else:
                    logger.error("Model does not have coef_ attribute. Cannot extract feature importance.")
                    return None
                
            else:
                logger.error(f"Unknown method: {method}. Please use 'random_forest', 'gradient_boosting', 'permutation', or 'lasso'.")
                return None
            
            # Sort by importance and store result
            importance_df = importance_df.sort_values('Importance', ascending=False)
            self.feature_importance = importance_df
            
            # Log top features
            top_features = importance_df.head(10)['Feature'].tolist()
            logger.info(f"Top 10 features: {', '.join(top_features)}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error in get_feature_importance: {e}")
            return None
    
    def plot_feature_importance(self, n_features=20, figsize=(12, 8), save_path=None):
        """
        Plot feature importance as a bar chart.
        
        Args:
            n_features (int, optional): Number of top features to include in the plot.
            figsize (tuple, optional): Figure size.
            save_path (str, optional): Path to save the figure. If None, the figure is not saved.
            
        Returns:
            matplotlib.figure.Figure: The created figure object.
        """
        if self.feature_importance is None:
            logger.error("No feature importance data available. Run get_feature_importance() first.")
            return None
        
        try:
            # Get top n features
            top_features = self.feature_importance.head(n_features).copy()
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Create color gradient based on importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            
            # Plot horizontal bar chart
            bars = plt.barh(
                y=range(len(top_features)),
                width=top_features['Importance'],
                color=colors
            )
            
            # Add feature names as y-tick labels
            plt.yticks(range(len(top_features)), top_features['Feature'])
            
            # Add values on bars for features with non-zero importance
            for i, bar in enumerate(bars):
                if top_features['Importance'].iloc[i] > 0:
                    plt.text(
                        bar.get_width() + bar.get_width() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{top_features['Importance'].iloc[i]:.4f}",
                        va='center'
                    )
            
            # Add title and labels
            plt.title(f'Top {n_features} Feature Importance', fontsize=15)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error in plot_feature_importance: {e}")
            return None
    
    def select_from_model(self, X, y, estimator=None, threshold='mean', prefit=False):
        """
        Select features using a trained model's feature importance.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            estimator (object, optional): Trained estimator or estimator object.
            threshold (str or float, optional): Threshold for selecting features.
                Options: 'mean', 'median', or a float.
            prefit (bool, optional): Whether the estimator is already fitted.
            
        Returns:
            pandas.DataFrame: DataFrame with only the selected features.
        """
        logger.info(f"Selecting features from model with threshold: {threshold}")
        
        # Default estimator if none provided
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            logger.info("Using default RandomForestClassifier for feature selection")
            prefit = False
        
        try:
            # Initialize selector
            selector = SelectFromModel(estimator=estimator, threshold=threshold, prefit=prefit)
            
            # Fit selector if estimator is not already fitted
            if not prefit:
                selector.fit(X, y)
            
            # Transform data
            X_new = selector.transform(X)
            
            # Get selected feature names if input is DataFrame
            if isinstance(X, pd.DataFrame):
                selected_features = X.columns[selector.get_support()]
                self.selected_features = selected_features
                
                # Log selected features
                logger.info(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
                
                return X[selected_features]
            else:
                self.selected_features = np.arange(X.shape[1])[selector.get_support()]
                logger.info(f"Selected {len(self.selected_features)} feature indices: {self.selected_features}")
                return X_new
                
        except Exception as e:
            logger.error(f"Error in select_from_model: {e}")
            return X
    
    def get_best_feature_subset(self, X, y, estimator=None, cv=5, scoring='roc_auc', max_features=None):
        """
        Find the optimal feature subset by evaluating model performance with different feature counts.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            estimator (object, optional): Estimator to use for evaluation.
            cv (int, optional): Number of cross-validation folds.
            scoring (str, optional): Scoring metric for evaluation.
            max_features (int, optional): Maximum number of features to consider.
                If None, will try up to min(30, n_features).
            
        Returns:
            tuple: (best_features, performance_df, best_score)
                - best_features: List of selected feature names
                - performance_df: DataFrame with performance for each feature count
                - best_score: Best cross-validation score
        """
        logger.info("Finding optimal feature subset")
        
        if not isinstance(X, pd.DataFrame):
            logger.warning("X is not a pandas DataFrame. Converting to DataFrame with default column names.")
            X = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
        
        # Default estimator if none provided
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            logger.info("Using default RandomForestClassifier for feature subset evaluation")
        
        # Set maximum number of features to try
        if max_features is None:
            max_features = min(30, X.shape[1])
        
        try:
            # Get feature importance
            if self.feature_importance is None:
                self.get_feature_importance(X, y)
            
            # Prepare storage for results
            performance_results = []
            all_features = self.feature_importance['Feature'].tolist()
            
            # Evaluate different feature subsets
            for n_features in range(1, max_features + 1):
                # Get top n features
                top_features = all_features[:n_features]
                
                # Prepare data with selected features
                X_selected = X[top_features]
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    estimator, X_selected, y,
                    cv=cv, scoring=scoring
                )
                
                # Store results
                performance_results.append({
                    'n_features': n_features,
                    'features': top_features,
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std()
                })
                
                logger.info(f"Evaluated {n_features} features: mean {scoring} = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}")
            
            # Convert results to DataFrame
            performance_df = pd.DataFrame(performance_results)
            
            # Find optimal number of features
            best_idx = performance_df['mean_score'].idxmax()
            best_n_features = performance_df.loc[best_idx, 'n_features']
            best_score = performance_df.loc[best_idx, 'mean_score']
            best_features = all_features[:best_n_features]
            
            logger.info(f"Optimal feature subset contains {best_n_features} features with {scoring} = {best_score:.4f}")
            
            # Store selected features
            self.selected_features = best_features
            
            return best_features, performance_df, best_score
            
        except Exception as e:
            logger.error(f"Error in get_best_feature_subset: {e}")
            return None, None, None
    
    def create_correlation_heatmap(self, X, figsize=(14, 12), mask_upper=True, save_path=None):
        """
        Create a correlation heatmap of features.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            figsize (tuple, optional): Figure size.
            mask_upper (bool, optional): Whether to mask the upper triangle of the heatmap.
            save_path (str, optional): Path to save the figure. If None, the figure is not saved.
            
        Returns:
            matplotlib.figure.Figure: The created figure object.
        """
        logger.info("Creating correlation heatmap")
        
        if not isinstance(X, pd.DataFrame):
            logger.warning("X is not a pandas DataFrame. Cannot create correlation heatmap.")
            return None
        
        try:
            # Compute correlation matrix
            corr_matrix = X.corr()
            
            # Create figure
            plt.figure(figsize=figsize)
            
            # Create mask for upper triangle
            mask = None
            if mask_upper:
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                annot=False,
                fmt='.2f',
                cbar_kws={'shrink': 0.8}
            )
            
            # Add title
            plt.title('Feature Correlation Matrix', fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure if path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation heatmap saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error in create_correlation_heatmap: {e}")
            return None
    
    def remove_highly_correlated(self, X, threshold=0.85):
        """
        Remove highly correlated features based on a correlation threshold.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            threshold (float, optional): Correlation threshold. Features with correlation above
                this threshold will be considered for removal.
            
        Returns:
            pandas.DataFrame: DataFrame with correlated features removed.
        """
        logger.info(f"Removing highly correlated features with threshold {threshold}")
        
        if not isinstance(X, pd.DataFrame):
            logger.warning("X is not a pandas DataFrame. Cannot remove correlated features.")
            return X
        
        try:
            # Compute correlation matrix
            corr_matrix = X.corr().abs()
            
            # Create upper triangle mask
            upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            
            # Find features with correlation above threshold
            to_drop = [column for column in corr_matrix.columns if any(corr_matrix.loc[column, upper] > threshold)]
            
            if len(to_drop) > 0:
                logger.info(f"Removing {len(to_drop)} highly correlated features: {', '.join(to_drop)}")
                
                # Remove highly correlated features
                X_filtered = X.drop(columns=to_drop)
                
                # Update selected features if they exist
                if self.selected_features is not None:
                    self.selected_features = [feature for feature in self.selected_features if feature not in to_drop]
                
                return X_filtered
            else:
                logger.info("No highly correlated features found.")
                return X
                
        except Exception as e:
            logger.error(f"Error in remove_highly_correlated: {e}")
            return X
