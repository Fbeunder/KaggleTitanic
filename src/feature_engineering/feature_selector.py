# -*- coding: utf-8 -*-
"""
Feature Selector Module

This module is responsible for selecting the most important features for model training.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier


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
        self.selected_features = None
    
    def select_k_best(self, X, y, k=10):
        """
        Select k best features using ANOVA F-value between features and target.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            k (int, optional): Number of top features to select.
            
        Returns:
            pandas.DataFrame: DataFrame with only the selected features.
        """
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        
        # Get selected feature names if input is DataFrame
        if isinstance(X, pd.DataFrame):
            selected_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_indices]
            self.selected_features = selected_features
            return X[selected_features]
        else:
            self.selected_features = np.arange(X.shape[1])[selector.get_support()]
            return X_new
    
    def select_with_rfe(self, X, y, n_features_to_select=10):
        """
        Select features using Recursive Feature Elimination with a random forest classifier.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            n_features_to_select (int, optional): Number of features to select.
            
        Returns:
            pandas.DataFrame: DataFrame with only the selected features.
        """
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector.fit(X, y)
        
        # Get selected feature names if input is DataFrame
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.support_]
            self.selected_features = selected_features
            return X[selected_features]
        else:
            self.selected_features = np.arange(X.shape[1])[selector.support_]
            return selector.transform(X)
    
    def get_feature_importance(self, X, y):
        """
        Get feature importance using a random forest classifier.
        
        Args:
            X (pandas.DataFrame): Feature matrix.
            y (pandas.Series or numpy.ndarray): Target vector.
            
        Returns:
            pandas.DataFrame: DataFrame with features and their importance scores.
        """
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        estimator.fit(X, y)
        
        # Create feature importance DataFrame
        if isinstance(X, pd.DataFrame):
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': estimator.feature_importances_
            })
        else:
            importance_df = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
                'Importance': estimator.feature_importances_
            })
        
        return importance_df.sort_values('Importance', ascending=False)
