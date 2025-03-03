# -*- coding: utf-8 -*-
"""
Feature Creator Module

This module is responsible for creating new features from existing ones in the Titanic dataset.
It implements various feature engineering techniques to create potentially more predictive features.
"""

import pandas as pd
import numpy as np
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        logger.info("Initializing FeatureCreator")
    
    def create_title_feature(self, data, name_column='Name'):
        """
        Extract titles from passenger names and group them into categories.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            name_column (str, optional): Column containing passenger names.
            
        Returns:
            pandas.DataFrame: Dataset with added title feature.
        """
        logger.info("Creating title features from names")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        if name_column not in result.columns:
            logger.warning(f"Column {name_column} not found in the dataset. Skipping title feature creation.")
            return result
        
        # Extract title from name using regex
        result['Title'] = result[name_column].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Check for missing titles and log
        missing_titles = result['Title'].isnull().sum()
        if missing_titles > 0:
            logger.warning(f"Found {missing_titles} rows with missing titles. These will be labeled as 'Unknown'.")
            result['Title'].fillna('Unknown', inplace=True)
        
        # Group rare titles into categories
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
        result['Title'] = result['Title'].map(title_mapping)
        
        # Handle any new titles not in the mapping
        unknown_titles = result[~result['Title'].isin(list(set(title_mapping.values())))]['Title'].unique()
        if len(unknown_titles) > 0:
            logger.warning(f"Found unmapped titles: {unknown_titles}. These will be mapped to 'Other'.")
            result.loc[~result['Title'].isin(list(set(title_mapping.values()))), 'Title'] = 'Other'
        
        # One-hot encode Title
        title_dummies = pd.get_dummies(result['Title'], prefix='Title', drop_first=False)
        result = pd.concat([result, title_dummies], axis=1)
        
        # Log counts for each title category for data exploration
        title_counts = result['Title'].value_counts()
        logger.info(f"Title distribution: {title_counts.to_dict()}")
        
        return result
    
    def create_family_size_feature(self, data, siblings_column='SibSp', parents_column='Parch'):
        """
        Create family size and family type features by combining siblings/spouses and parents/children columns.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            siblings_column (str, optional): Column containing siblings/spouses count.
            parents_column (str, optional): Column containing parents/children count.
            
        Returns:
            pandas.DataFrame: Dataset with added family size and type features.
        """
        logger.info("Creating family size and type features")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Check if required columns exist
        if siblings_column not in result.columns or parents_column not in result.columns:
            logger.warning(f"Columns {siblings_column} or {parents_column} not found. Skipping family features creation.")
            return result
        
        # Family size is the sum of siblings/spouses, parents/children, and the passenger themselves
        result['FamilySize'] = result[siblings_column] + result[parents_column] + 1
        
        # Create IsAlone flag
        result['IsAlone'] = 0
        result.loc[result['FamilySize'] == 1, 'IsAlone'] = 1
        
        # Create family size categories
        def classify_family_size(size):
            if size == 1:
                return 'Alone'
            elif size <= 4:
                return 'Small'
            else:
                return 'Large'
        
        result['FamilyType'] = result['FamilySize'].apply(classify_family_size)
        
        # Create family survival rate feature - this requires the Survived column
        if 'Survived' in result.columns and 'PassengerId' not in result.columns:
            logger.warning("Cannot create family survival rate without PassengerId column.")
        elif 'Survived' in result.columns:
            # Create family identifiers
            result['FamilyID'] = result['LastName'] + '_' + result[siblings_column].astype(str) + '_' + result[parents_column].astype(str)
            
            # Calculate family survival rates
            family_survival = result.groupby('FamilyID')['Survived'].mean().reset_index()
            family_survival.columns = ['FamilyID', 'FamilySurvivalRate']
            
            # Merge back to the original dataset
            result = pd.merge(result, family_survival, on='FamilyID', how='left')
            
            # Fill NA values for singleton families
            result['FamilySurvivalRate'].fillna(result['Survived'], inplace=True)
            
            logger.info("Added family survival rate feature")
        
        # One-hot encode FamilyType
        family_type_dummies = pd.get_dummies(result['FamilyType'], prefix='FamilyType', drop_first=False)
        result = pd.concat([result, family_type_dummies], axis=1)
        
        # Log family size distribution
        family_size_counts = result['FamilySize'].value_counts().sort_index()
        logger.info(f"Family size distribution: {family_size_counts.to_dict()}")
        logger.info(f"Family type distribution: {result['FamilyType'].value_counts().to_dict()}")
        
        return result
    
    def create_cabin_features(self, data, cabin_column='Cabin'):
        """
        Create features from the cabin information including deck, cabin count, and cabin location.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            cabin_column (str, optional): Column containing cabin information.
            
        Returns:
            pandas.DataFrame: Dataset with added cabin features.
        """
        logger.info("Creating cabin features")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        if cabin_column not in result.columns:
            logger.warning(f"Column {cabin_column} not found. Skipping cabin features creation.")
            return result
        
        # Create HasCabin feature (0 if cabin is NA, 1 otherwise)
        result['HasCabin'] = result[cabin_column].notna().astype(int)
        
        # Extract cabin deck (first letter of the cabin)
        result['CabinDeck'] = result[cabin_column].str.slice(0, 1)
        
        # Fill missing values with 'U' for Unknown
        result['CabinDeck'].fillna('U', inplace=True)
        
        # Count number of cabins per passenger
        result['CabinCount'] = result[cabin_column].str.split().str.len()
        
        # Fill NA values with 0
        result['CabinCount'].fillna(0, inplace=True)
        
        # Extract cabin number and convert to numeric for approximately determining location
        def extract_cabin_number(cabin):
            if pd.isna(cabin):
                return np.nan
            try:
                # Extract digits from the first cabin
                cabin_number = re.search(r'(\d+)', cabin.split()[0])
                if cabin_number:
                    return int(cabin_number.group(1))
                else:
                    return np.nan
            except Exception as e:
                logger.error(f"Error extracting cabin number from {cabin}: {e}")
                return np.nan
        
        result['CabinNumber'] = result[cabin_column].apply(extract_cabin_number)
        
        # Categorize cabin location (bow, stern, midship) based on cabin number
        def categorize_cabin_location(number):
            if pd.isna(number):
                return 'Unknown'
            elif number < 30:
                return 'Bow'
            elif number < 70:
                return 'Midship'
            else:
                return 'Stern'
        
        result['CabinLocation'] = result['CabinNumber'].apply(categorize_cabin_location)
        
        # One-hot encode CabinDeck and CabinLocation
        deck_dummies = pd.get_dummies(result['CabinDeck'], prefix='Deck', drop_first=False)
        location_dummies = pd.get_dummies(result['CabinLocation'], prefix='CabinLocation', drop_first=False)
        
        result = pd.concat([result, deck_dummies, location_dummies], axis=1)
        
        # Log cabin feature statistics
        logger.info(f"Passengers with cabin information: {result['HasCabin'].sum()} ({result['HasCabin'].mean()*100:.2f}%)")
        logger.info(f"Cabin deck distribution: {result['CabinDeck'].value_counts().to_dict()}")
        logger.info(f"Cabin location distribution: {result['CabinLocation'].value_counts().to_dict()}")
        
        return result
    
    def create_age_categories(self, data, age_column='Age'):
        """
        Create age categories and related features.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            age_column (str, optional): Column containing age data.
            
        Returns:
            pandas.DataFrame: Dataset with added age category features.
        """
        logger.info("Creating age category features")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        if age_column not in result.columns:
            logger.warning(f"Column {age_column} not found. Skipping age category features creation.")
            return result
        
        # Handle missing age values if necessary
        missing_ages = result[age_column].isnull().sum()
        if missing_ages > 0:
            logger.warning(f"Found {missing_ages} rows with missing ages. Consider imputing before feature creation.")
        
        # Create age categories using bins
        age_bins = [0, 5, 12, 18, 30, 50, 65, 120]
        age_labels = ['Infant', 'Child', 'Teen', 'YoungAdult', 'Adult', 'Senior', 'Elderly']
        
        result['AgeCategory'] = pd.cut(
            result[age_column],
            bins=age_bins,
            labels=age_labels,
            right=False
        )
        
        # Fill missing values with 'Unknown'
        if missing_ages > 0:
            result['AgeCategory'].fillna('Unknown', inplace=True)
            age_labels.append('Unknown')
        
        # Create IsChild feature (1 if age < 18, 0 otherwise)
        result['IsChild'] = 0
        result.loc[result[age_column] < 18, 'IsChild'] = 1
        
        # Fill missing values based on title if available
        if 'Title' in result.columns:
            # Map titles to estimated child status (Master titles are typically for boys)
            result.loc[(result[age_column].isnull()) & (result['Title'] == 'Master'), 'IsChild'] = 1
        
        # One-hot encode AgeCategory
        age_dummies = pd.get_dummies(result['AgeCategory'], prefix='Age', drop_first=False)
        result = pd.concat([result, age_dummies], axis=1)
        
        # Log age category distribution
        age_counts = result['AgeCategory'].value_counts()
        logger.info(f"Age category distribution: {age_counts.to_dict()}")
        
        return result
    
    def create_fare_categories(self, data, fare_column='Fare'):
        """
        Create fare categories and related features.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            fare_column (str, optional): Column containing fare data.
            
        Returns:
            pandas.DataFrame: Dataset with added fare category features.
        """
        logger.info("Creating fare category features")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        if fare_column not in result.columns:
            logger.warning(f"Column {fare_column} not found. Skipping fare category features creation.")
            return result
        
        # Handle missing fare values if necessary
        missing_fares = result[fare_column].isnull().sum()
        if missing_fares > 0:
            logger.warning(f"Found {missing_fares} rows with missing fares. Consider imputing before feature creation.")
        
        # Create FarePerPerson feature if family size is available
        if 'FamilySize' in result.columns:
            result['FarePerPerson'] = result[fare_column] / result['FamilySize']
            logger.info("Created FarePerPerson feature")
        
        # Create fare categories using qcut for equal-sized bins
        result['FareCategory'] = pd.qcut(
            result[fare_column],
            q=5,
            labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
        )
        
        # Fill missing values
        if missing_fares > 0:
            result['FareCategory'].fillna('Unknown', inplace=True)
        
        # Create logarithmic fare feature to handle skewness
        result['LogFare'] = np.log1p(result[fare_column])
        
        # Create FareRatio features (fare relative to class)
        if 'Pclass' in result.columns:
            # Calculate average fare per class
            class_avg_fares = result.groupby('Pclass')[fare_column].transform('mean')
            result['FareRatio'] = result[fare_column] / class_avg_fares
            logger.info("Created FareRatio feature")
        
        # One-hot encode FareCategory
        fare_dummies = pd.get_dummies(result['FareCategory'], prefix='Fare', drop_first=False)
        result = pd.concat([result, fare_dummies], axis=1)
        
        # Log fare category distribution
        fare_counts = result['FareCategory'].value_counts()
        logger.info(f"Fare category distribution: {fare_counts.to_dict()}")
        
        return result
    
    def create_embarked_features(self, data, embarked_column='Embarked'):
        """
        Create embarked-related features.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            embarked_column (str, optional): Column containing embarked data.
            
        Returns:
            pandas.DataFrame: Dataset with added embarked-related features.
        """
        logger.info("Creating embarked features")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        if embarked_column not in result.columns:
            logger.warning(f"Column {embarked_column} not found. Skipping embarked features creation.")
            return result
        
        # Handle missing embarked values
        missing_embarked = result[embarked_column].isnull().sum()
        if missing_embarked > 0:
            logger.warning(f"Found {missing_embarked} rows with missing embarked. These will be handled during encoding.")
        
        # Create EmbarkedSurvival feature if Survived column is available
        if 'Survived' in result.columns and not all(result['Survived'].isnull()):
            # Calculate survival rate per embarked port
            embarked_survival = result.groupby(embarked_column)['Survived'].transform('mean')
            result['EmbarkedSurvivalRate'] = embarked_survival
            logger.info("Created EmbarkedSurvivalRate feature")
        
        # Port specific features
        # Create features based on geographical/historical knowledge of ports
        port_distances = {
            'C': 0,     # Cherbourg - closest to England
            'Q': 1,     # Queenstown - middle distance
            'S': 2,     # Southampton - furthest from America
        }
        
        result['PortDistance'] = result[embarked_column].map(port_distances)
        
        # Fill NA values with median port distance
        median_distance = np.median(list(port_distances.values()))
        result['PortDistance'].fillna(median_distance, inplace=True)
        
        # Log embarked feature statistics
        logger.info(f"Embarked distribution: {result[embarked_column].value_counts().to_dict()}")
        if 'Survived' in result.columns and not all(result['Survived'].isnull()):
            survival_by_port = result.groupby(embarked_column)['Survived'].mean()
            logger.info(f"Survival rates by port: {survival_by_port.to_dict()}")
        
        return result
    
    def combine_features(self, data):
        """
        Create feature interactions by combining existing features.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            
        Returns:
            pandas.DataFrame: Dataset with added feature interaction columns.
        """
        logger.info("Creating feature interactions")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Define feature combinations to create
        feature_combinations = []
        
        # Pclass and Gender interaction
        if 'Pclass' in result.columns and 'Sex' in result.columns:
            result['Pclass_Sex'] = result['Pclass'].astype(str) + '_' + result['Sex'].astype(str)
            feature_combinations.append('Pclass_Sex')
            logger.info("Created Pclass_Sex interaction feature")
        
        # Age and Class interaction
        if 'IsChild' in result.columns and 'Pclass' in result.columns:
            result['IsChild_Pclass'] = result['IsChild'].astype(str) + '_' + result['Pclass'].astype(str)
            feature_combinations.append('IsChild_Pclass')
            logger.info("Created IsChild_Pclass interaction feature")
        
        # Family and Class interaction
        if 'FamilyType' in result.columns and 'Pclass' in result.columns:
            result['FamilyType_Pclass'] = result['FamilyType'].astype(str) + '_' + result['Pclass'].astype(str)
            feature_combinations.append('FamilyType_Pclass')
            logger.info("Created FamilyType_Pclass interaction feature")
        
        # Cabin and Class interaction
        if 'HasCabin' in result.columns and 'Pclass' in result.columns:
            result['HasCabin_Pclass'] = result['HasCabin'].astype(str) + '_' + result['Pclass'].astype(str)
            feature_combinations.append('HasCabin_Pclass')
            logger.info("Created HasCabin_Pclass interaction feature")
        
        # One-hot encode all interaction features
        for feature in feature_combinations:
            interaction_dummies = pd.get_dummies(result[feature], prefix=feature, drop_first=False)
            result = pd.concat([result, interaction_dummies], axis=1)
            # Drop the original string concatenation column
            result.drop(feature, axis=1, inplace=True)
        
        # Create some mathematical interactions
        if 'Age' in result.columns and 'FamilySize' in result.columns:
            result['Age_x_FamilySize'] = result['Age'] * result['FamilySize']
            logger.info("Created Age_x_FamilySize numerical interaction feature")
        
        if 'Fare' in result.columns and 'FamilySize' in result.columns:
            result['Fare_x_FamilySize'] = result['Fare'] * result['FamilySize']
            logger.info("Created Fare_x_FamilySize numerical interaction feature")
        
        if 'Age' in result.columns and 'Fare' in result.columns:
            result['Age_Fare_Ratio'] = result['Age'] / (result['Fare'] + 1)  # Add 1 to avoid division by zero
            logger.info("Created Age_Fare_Ratio numerical interaction feature")
        
        return result
    
    def create_all_features(self, data):
        """
        Convenience method to create all features in sequence.
        
        Args:
            data (pandas.DataFrame): Dataset containing passenger information.
            
        Returns:
            pandas.DataFrame: Dataset with all additional features.
        """
        logger.info("Creating all features")
        
        result = data.copy()
        
        # Extract additional features from Name
        result = self.create_title_feature(result)
        
        # Create family-related features
        result = self.create_family_size_feature(result)
        
        # Create cabin-related features
        result = self.create_cabin_features(result)
        
        # Create age-related features
        result = self.create_age_categories(result)
        
        # Create fare-related features
        result = self.create_fare_categories(result)
        
        # Create embarked-related features
        result = self.create_embarked_features(result)
        
        # Create feature interactions
        result = self.combine_features(result)
        
        # Log final feature count
        initial_cols = data.shape[1]
        final_cols = result.shape[1]
        logger.info(f"Feature creation completed. Initial columns: {initial_cols}, Final columns: {final_cols}, New features: {final_cols - initial_cols}")
        
        return result
