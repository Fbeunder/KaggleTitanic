# -*- coding: utf-8 -*-
"""
Data Loader Module

This module is responsible for loading the Titanic dataset from CSV files.
It provides functions to load training and testing data.
"""

import pandas as pd
import os
import logging
from src.utilities.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.data_dir = data_dir if data_dir else self.config.get('data_dir', '')
        
        # If data_dir is empty, use the default path
        if not self.data_dir:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            
        # Check if data directory exists, if not try to use current directory
        if not os.path.exists(self.data_dir):
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if os.path.exists(os.path.join(root_dir, 'train.csv')):
                self.data_dir = root_dir
                logger.info(f"Data directory not found. Using root directory: {self.data_dir}")
            else:
                logger.warning(f"Data directory not found: {self.data_dir}. Using current directory.")
                self.data_dir = os.getcwd()
    
    def load_train_data(self):
        """
        Load the training dataset.
        
        Returns:
            pandas.DataFrame: The training dataset.
            
        Raises:
            FileNotFoundError: If the training data file is not found.
            pd.errors.EmptyDataError: If the training data file is empty.
            pd.errors.ParserError: If the training data file cannot be parsed.
        """
        try:
            file_path = os.path.join(self.data_dir, 'train.csv')
            logger.info(f"Loading training data from {file_path}")
            
            if not os.path.exists(file_path):
                error_msg = f"Training data file not found: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded training data with shape {data.shape}")
            return data
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {e}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading training data: {e}")
            raise
    
    def load_test_data(self):
        """
        Load the testing dataset.
        
        Returns:
            pandas.DataFrame: The testing dataset.
            
        Raises:
            FileNotFoundError: If the testing data file is not found.
            pd.errors.EmptyDataError: If the testing data file is empty.
            pd.errors.ParserError: If the testing data file cannot be parsed.
        """
        try:
            file_path = os.path.join(self.data_dir, 'test.csv')
            logger.info(f"Loading testing data from {file_path}")
            
            if not os.path.exists(file_path):
                error_msg = f"Testing data file not found: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded testing data with shape {data.shape}")
            return data
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {e}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading testing data: {e}")
            raise
    
    def get_combined_data(self):
        """
        Load and combine both training and testing datasets.
        This is useful for applying consistent preprocessing.
        
        Returns:
            tuple: A tuple containing:
                - pandas.DataFrame: Combined data with a 'is_train' column
                - int: Number of training samples
        
        Raises:
            Exception: If either dataset fails to load
        """
        try:
            train_data = self.load_train_data()
            test_data = self.load_test_data()
            
            # Add indicator column
            train_data['is_train'] = 1
            test_data['is_train'] = 0
            
            # Combine datasets
            combined_data = pd.concat([train_data, test_data], axis=0, sort=False)
            
            # Reset index to avoid duplicates
            combined_data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Successfully combined data with shape {combined_data.shape}")
            return combined_data, len(train_data)
            
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            raise
