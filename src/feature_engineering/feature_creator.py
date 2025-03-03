# -*- coding: utf-8 -*-
"""
Feature Creator Module

This module is responsible for creating new features from existing ones in the Titanic dataset.
"""

import pandas as pd
import numpy as np


class FeatureCreator:
    """
    FeatureCreator class for creating new features from existing ones.
    
    This class implements various feature engineering techniques to create
    new, potentially more predictive features for the Titanic dataset.
    """
    
    def __init__(self):
        """
        Initialize the FeatureCreator.
        """
        pass
    
    def create_title_feature(self, data, name_column='Name'):
        """
        Extract titles from passenger names.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            name_column (str, optional): Column containing passenger names.
            
        Returns:
            pandas.DataFrame: Dataset with added title feature.
        """
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # This would extract titles like Mr., Mrs., Miss., etc.
        # For now, this is just a placeholder
        result['Title'] = 'Mr'  # Placeholder
        
        return result
    
    def create_family_size_feature(self, data, siblings_column='SibSp', parents_column='Parch'):
        """
        Create a family size feature by combining siblings/spouses and parents/children columns.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            siblings_column (str, optional): Column containing siblings/spouses count.
            parents_column (str, optional): Column containing parents/children count.
            
        Returns:
            pandas.DataFrame: Dataset with added family size feature.
        """
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Family size is the sum of siblings/spouses, parents/children, and the passenger themselves
        if siblings_column in result.columns and parents_column in result.columns:
            result['FamilySize'] = result[siblings_column] + result[parents_column] + 1
        
        return result
    
    def create_cabin_features(self, data, cabin_column='Cabin'):
        """
        Create features from the cabin information.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            cabin_column (str, optional): Column containing cabin information.
            
        Returns:
            pandas.DataFrame: Dataset with added cabin features.
        """
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # This would extract cabin deck, cabin number, etc.
        # For now, this is just a placeholder
        if cabin_column in result.columns:
            result['HasCabin'] = result[cabin_column].notna().astype(int)
        
        return result
