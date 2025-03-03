# -*- coding: utf-8 -*-
"""
Data Preprocessor Module

This module is responsible for preprocessing the Titanic dataset, including
cleaning, feature encoding, and handling missing values.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    DataPreprocessor class for preprocessing Titanic datasets.
    
    This class handles data cleaning, transformation, encoding,
    and missing value imputation for the Titanic dataset.
    """
    
    def __init__(self):
        """
        Initialize the DataPreprocessor.
        """
        self.numeric_features = ['Age', 'Fare']
        self.categorical_features = ['Pclass', 'Sex', 'Embarked']
        self.drop_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'is_train']
        
        # The preprocessors will be initialized during fit_transform
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}
        self.preprocessing_pipeline = None
        
        # Store column names for transformed data
        self.transformed_columns = None
    
    def handle_missing_values(self, data):
        """
        Handle missing values in the dataset.
        
        Args:
            data (pandas.DataFrame): Dataset with missing values.
            
        Returns:
            pandas.DataFrame: Dataset with handled missing values.
        """
        logger.info("Handling missing values")
        data_copy = data.copy()
        
        # Check missing values
        missing_values = data_copy.isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_values[missing_values > 0]}")
        
        # Handle Age: impute with median
        if 'Age' in data_copy.columns and data_copy['Age'].isnull().any():
            self.numeric_imputer = SimpleImputer(strategy='median')
            data_copy['Age'] = self.numeric_imputer.fit_transform(data_copy[['Age']])
            
        # Handle Embarked: impute with most frequent
        if 'Embarked' in data_copy.columns and data_copy['Embarked'].isnull().any():
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            data_copy['Embarked'] = self.categorical_imputer.fit_transform(data_copy[['Embarked']])
            
        # Handle Fare: impute with median
        if 'Fare' in data_copy.columns and data_copy['Fare'].isnull().any():
            if self.numeric_imputer is None:
                self.numeric_imputer = SimpleImputer(strategy='median')
            data_copy['Fare'] = self.numeric_imputer.fit_transform(data_copy[['Fare']])
        
        # Log missing values after imputation
        missing_after = data_copy.isnull().sum()
        logger.info(f"Missing values after imputation:\n{missing_after[missing_after > 0]}")
        
        return data_copy
    
    def encode_categorical(self, data):
        """
        Encode categorical features in the dataset.
        
        Args:
            data (pandas.DataFrame): Dataset with categorical features.
            
        Returns:
            pandas.DataFrame: Dataset with encoded categorical features.
        """
        logger.info("Encoding categorical features")
        data_copy = data.copy()
        
        # Encode Sex using LabelEncoder
        if 'Sex' in data_copy.columns:
            self.encoders['Sex'] = LabelEncoder()
            data_copy['Sex'] = self.encoders['Sex'].fit_transform(data_copy['Sex'])
            logger.info(f"Sex encoding mapping: {dict(zip(self.encoders['Sex'].classes_, range(len(self.encoders['Sex'].classes_))))}")
        
        # One-hot encode Embarked
        if 'Embarked' in data_copy.columns:
            # Handle missing values first to prevent errors
            if data_copy['Embarked'].isnull().any():
                data_copy['Embarked'].fillna(data_copy['Embarked'].mode()[0], inplace=True)
                
            # Apply one-hot encoding
            embarked_dummies = pd.get_dummies(data_copy['Embarked'], prefix='Embarked', drop_first=False)
            data_copy = pd.concat([data_copy, embarked_dummies], axis=1)
            data_copy.drop('Embarked', axis=1, inplace=True)
            logger.info(f"Created {embarked_dummies.shape[1]} one-hot features for Embarked")
        
        # Pclass can be treated as ordinal or one-hot encoded
        if 'Pclass' in data_copy.columns:
            # Convert to one-hot
            pclass_dummies = pd.get_dummies(data_copy['Pclass'], prefix='Pclass', drop_first=False)
            data_copy = pd.concat([data_copy, pclass_dummies], axis=1)
            data_copy.drop('Pclass', axis=1, inplace=True)
            logger.info(f"Created {pclass_dummies.shape[1]} one-hot features for Pclass")
        
        return data_copy
    
    def scale_numeric(self, data):
        """
        Scale numeric features in the dataset.
        
        Args:
            data (pandas.DataFrame): Dataset with numeric features.
            
        Returns:
            pandas.DataFrame: Dataset with scaled numeric features.
        """
        logger.info("Scaling numeric features")
        data_copy = data.copy()
        
        # Get available numeric features (some might be missing in the test set)
        numeric_cols = [col for col in self.numeric_features if col in data_copy.columns]
        
        if len(numeric_cols) > 0:
            self.scaler = StandardScaler()
            data_copy[numeric_cols] = self.scaler.fit_transform(data_copy[numeric_cols])
            logger.info(f"Scaled {len(numeric_cols)} numeric features")
        
        return data_copy
    
    def extract_title(self, data):
        """
        Extract title from Name column as a new feature.
        
        Args:
            data (pandas.DataFrame): Dataset with Name column.
            
        Returns:
            pandas.DataFrame: Dataset with Title column added.
        """
        logger.info("Extracting title from Name")
        data_copy = data.copy()
        
        if 'Name' in data_copy.columns:
            # Extract title from name
            data_copy['Title'] = data_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            
            # Group rare titles
            title_mapping = {
                'Mr': 'Mr',
                'Miss': 'Miss',
                'Mrs': 'Mrs',
                'Master': 'Master',
                'Dr': 'Rare',
                'Rev': 'Rare',
                'Col': 'Rare',
                'Major': 'Rare',
                'Mlle': 'Miss',
                'Countess': 'Rare',
                'Ms': 'Miss',
                'Lady': 'Rare',
                'Jonkheer': 'Rare',
                'Don': 'Rare',
                'Dona': 'Rare',
                'Mme': 'Mrs',
                'Capt': 'Rare',
                'Sir': 'Rare'
            }
            
            # Map titles to groups
            data_copy['Title'] = data_copy['Title'].map(title_mapping)
            
            # Fill any missing titles with 'Unknown'
            data_copy['Title'].fillna('Unknown', inplace=True)
            
            # Encode Title
            title_dummies = pd.get_dummies(data_copy['Title'], prefix='Title', drop_first=False)
            data_copy = pd.concat([data_copy, title_dummies], axis=1)
            data_copy.drop('Title', axis=1, inplace=True)
            
            logger.info(f"Created {title_dummies.shape[1]} one-hot features for Title")
        
        return data_copy
    
    def create_family_features(self, data):
        """
        Create features related to family size and structure.
        
        Args:
            data (pandas.DataFrame): Dataset with SibSp and Parch columns.
            
        Returns:
            pandas.DataFrame: Dataset with family features added.
        """
        logger.info("Creating family features")
        data_copy = data.copy()
        
        if 'SibSp' in data_copy.columns and 'Parch' in data_copy.columns:
            # Family size
            data_copy['FamilySize'] = data_copy['SibSp'] + data_copy['Parch'] + 1
            
            # Family group
            data_copy['IsAlone'] = 0
            data_copy.loc[data_copy['FamilySize'] == 1, 'IsAlone'] = 1
            
            # Family size categories
            data_copy['FamilyGroup'] = pd.cut(
                data_copy['FamilySize'],
                bins=[0, 1, 4, float('inf')],
                labels=['Alone', 'Small', 'Large']
            )
            
            # One-hot encode FamilyGroup
            family_dummies = pd.get_dummies(data_copy['FamilyGroup'], prefix='FamilyGroup', drop_first=False)
            data_copy = pd.concat([data_copy, family_dummies], axis=1)
            data_copy.drop('FamilyGroup', axis=1, inplace=True)
            
            logger.info("Created family features: FamilySize, IsAlone, and FamilyGroup one-hot features")
        
        return data_copy
    
    def fit_transform(self, data):
        """
        Fit preprocessors to the data and transform it.
        
        Args:
            data (pandas.DataFrame): Raw dataset to preprocess.
            
        Returns:
            pandas.DataFrame: Preprocessed dataset.
        """
        logger.info("Starting fit_transform on data")
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Extract features before dropping columns
        processed_data = self.extract_title(processed_data)
        processed_data = self.create_family_features(processed_data)
        
        # Handle missing values
        processed_data = self.handle_missing_values(processed_data)
        
        # Encode categorical features
        processed_data = self.encode_categorical(processed_data)
        
        # Scale numeric features
        processed_data = self.scale_numeric(processed_data)
        
        # Drop unnecessary columns
        drop_cols = [col for col in self.drop_features if col in processed_data.columns]
        processed_data.drop(drop_cols, axis=1, inplace=True)
        
        # Store the column names for future transforms
        self.transformed_columns = processed_data.columns.tolist()
        
        logger.info(f"Completed fit_transform. Final shape: {processed_data.shape}")
        logger.info(f"Final columns: {self.transformed_columns}")
        
        return processed_data
    
    def transform(self, data):
        """
        Transform data using fitted preprocessors.
        
        Args:
            data (pandas.DataFrame): Raw dataset to preprocess.
            
        Returns:
            pandas.DataFrame: Preprocessed dataset.
        """
        logger.info("Starting transform on data")
        
        if not self.transformed_columns:
            logger.warning("transform() called before fit_transform(). Running fit_transform() instead.")
            return self.fit_transform(data)
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Extract features before dropping columns
        processed_data = self.extract_title(processed_data)
        processed_data = self.create_family_features(processed_data)
        
        # Handle missing values using already fitted imputers
        if self.numeric_imputer is not None and 'Age' in processed_data.columns:
            processed_data['Age'] = self.numeric_imputer.transform(processed_data[['Age']])
        
        if self.numeric_imputer is not None and 'Fare' in processed_data.columns and processed_data['Fare'].isnull().any():
            processed_data['Fare'] = self.numeric_imputer.transform(processed_data[['Fare']])
        
        if self.categorical_imputer is not None and 'Embarked' in processed_data.columns and processed_data['Embarked'].isnull().any():
            processed_data['Embarked'] = self.categorical_imputer.transform(processed_data[['Embarked']])
        
        # Apply categorical encoding
        if 'Sex' in processed_data.columns and 'Sex' in self.encoders:
            processed_data['Sex'] = self.encoders['Sex'].transform(processed_data['Sex'])
        
        # Apply one-hot encoding for Embarked
        if 'Embarked' in processed_data.columns:
            embarked_dummies = pd.get_dummies(processed_data['Embarked'], prefix='Embarked', drop_first=False)
            processed_data = pd.concat([processed_data, embarked_dummies], axis=1)
            processed_data.drop('Embarked', axis=1, inplace=True)
        
        # Apply one-hot encoding for Pclass
        if 'Pclass' in processed_data.columns:
            pclass_dummies = pd.get_dummies(processed_data['Pclass'], prefix='Pclass', drop_first=False)
            processed_data = pd.concat([processed_data, pclass_dummies], axis=1)
            processed_data.drop('Pclass', axis=1, inplace=True)
        
        # Scale numeric features
        if self.scaler is not None:
            numeric_cols = [col for col in self.numeric_features if col in processed_data.columns]
            if len(numeric_cols) > 0:
                processed_data[numeric_cols] = self.scaler.transform(processed_data[numeric_cols])
        
        # Drop unnecessary columns
        drop_cols = [col for col in self.drop_features if col in processed_data.columns]
        processed_data.drop(drop_cols, axis=1, inplace=True)
        
        # Ensure all columns from fit_transform are present
        for col in self.transformed_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0
                logger.warning(f"Adding missing column {col} with zeros")
        
        # Ensure columns are in the same order
        processed_data = processed_data[self.transformed_columns]
        
        logger.info(f"Completed transform. Final shape: {processed_data.shape}")
        
        return processed_data
