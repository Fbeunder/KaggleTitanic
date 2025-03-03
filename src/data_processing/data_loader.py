# -*- coding: utf-8 -*-
"""
Data Loader Module

This module is responsible for loading the Titanic dataset from CSV files.
It provides functions to load training and testing data.
"""

import pandas as pd
import os
from src.utilities.config import get_config


class DataLoader:
    """
    DataLoader class for loading Titanic datasets.
    
    This class handles the loading of training and testing datasets for the 
    Titanic survival prediction project.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir (str, optional): Directory containing the data files.
                                     If None, uses the directory from config.
        """
        self.config = get_config()
        self.data_dir = data_dir if data_dir else self.config.get('data_dir', 'data')
    
    def load_train_data(self):
        """
        Load the training dataset.
        
        Returns:
            pandas.DataFrame: The training dataset.
        """
        try:
            file_path = os.path.join(self.data_dir, 'train.csv')
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading training data: {e}")
            return None
    
    def load_test_data(self):
        """
        Load the testing dataset.
        
        Returns:
            pandas.DataFrame: The testing dataset.
        """
        try:
            file_path = os.path.join(self.data_dir, 'test.csv')
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading testing data: {e}")
            return None
