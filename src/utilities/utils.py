# -*- coding: utf-8 -*-
"""
Utilities Module

This module provides utility functions for the Titanic survival prediction project.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_directory_if_not_exists(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): Path to the directory to create.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def get_project_root():
    """
    Get the root directory of the project.
    
    Returns:
        str: Path to the project root directory.
    """
    # This assumes that this module is in src/utilities
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def save_dataframe(df, filename, directory='data'):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        filename (str): Name of the file.
        directory (str, optional): Directory to save the file to.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        full_path = os.path.join(get_project_root(), directory, filename)
        create_directory_if_not_exists(os.path.dirname(full_path))
        df.to_csv(full_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving DataFrame to {full_path}: {e}")
        return False


def plot_correlation_matrix(df, figsize=(10, 8)):
    """
    Plot a correlation matrix for the numeric columns in a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to plot correlation matrix for.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Correlation Matrix')
    
    return fig


def print_dataframe_info(df):
    """
    Print information about a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to print information about.
    """
    print(f"DataFrame Shape: {df.shape}")
    print("\nDataFrame Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
