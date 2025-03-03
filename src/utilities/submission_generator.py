# -*- coding: utf-8 -*-
"""
Submission Generator Module

This module generates submission files for the Kaggle Titanic competition.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.data_processing.data_loader import DataLoader
from src.utilities.utils import create_directory_if_not_exists, get_project_root
from src.utilities.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubmissionGenerator:
    """
    SubmissionGenerator class for creating Kaggle submission files.
    
    This class generates submission files in the correct format
    for the Kaggle Titanic competition.
    """
    
    def __init__(self):
        """
        Initialize the SubmissionGenerator.
        """
        self.config = get_config()
        self.submissions_dir = self.config.get('submissions_dir', 'submissions')
        self.full_submissions_dir = os.path.join(get_project_root(), self.submissions_dir)
        create_directory_if_not_exists(self.full_submissions_dir)
    
    def generate_submission(self, model, X_test, file_name=None):
        """
        Generate a submission file for the Kaggle competition.
        
        Args:
            model: Trained model to use for predictions.
            X_test (pandas.DataFrame): Test data features.
            file_name (str, optional): Name of the submission file.
                                     If None, a default name is generated.
                                     
        Returns:
            str: Path to the generated submission file.
        """
        # Make predictions
        predictions = model.predict(X_test)
        
        # Load the original test data to get passenger IDs
        loader = DataLoader()
        test_data = loader.load_test_data()
        
        if test_data is None:
            raise ValueError("Could not load test data.")
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': test_data['PassengerId'],
            'Survived': predictions
        })
        
        # Generate file name if not provided
        if file_name is None:
            model_name = model.__class__.__name__
            file_name = f"submission_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Make sure file has .csv extension
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        
        # Save submission
        submission_path = os.path.join(self.full_submissions_dir, file_name)
        submission.to_csv(submission_path, index=False)
        
        logger.info(f"Submission saved to {submission_path}")
        return submission_path
    
    def validate_submission(self, submission_path):
        """
        Validate if the submission file meets Kaggle requirements.
        
        Args:
            submission_path (str): Path to the submission file.
            
        Returns:
            dict: Validation results with success flag and any issues found.
        """
        try:
            # Check if file exists
            if not os.path.exists(submission_path):
                return {
                    'valid': False,
                    'issues': ['Submission file does not exist.']
                }
            
            # Load the submission file
            submission = pd.read_csv(submission_path)
            
            # Check for required columns
            issues = []
            
            if 'PassengerId' not in submission.columns:
                issues.append('Missing PassengerId column.')
            
            if 'Survived' not in submission.columns:
                issues.append('Missing Survived column.')
            
            # Check if PassengerId is a unique identifier
            if 'PassengerId' in submission.columns and submission['PassengerId'].duplicated().any():
                issues.append('PassengerId column contains duplicate values.')
            
            # Check if Survived column contains only valid values (0 or 1)
            if 'Survived' in submission.columns:
                valid_values = set([0, 1])
                actual_values = set(submission['Survived'].unique())
                invalid_values = actual_values - valid_values
                if invalid_values:
                    issues.append(f'Survived column contains invalid values: {invalid_values}')
            
            # Check if submission has the expected number of rows
            loader = DataLoader()
            test_data = loader.load_test_data()
            if test_data is not None and len(submission) != len(test_data):
                issues.append(f'Submission has {len(submission)} rows, but test data has {len(test_data)} rows.')
            
            # Return validation results
            return {
                'valid': len(issues) == 0,
                'issues': issues if len(issues) > 0 else None
            }
        except Exception as e:
            return {
                'valid': False,
                'issues': [f'Error validating submission: {str(e)}']
            }
    
    def compare_submissions(self, submission_paths):
        """
        Compare multiple submission files to identify differences.
        
        Args:
            submission_paths (list): List of paths to submission files.
            
        Returns:
            dict: Comparison results including differences and statistics.
        """
        if not submission_paths or len(submission_paths) < 2:
            return {
                'error': 'Need at least two submission files to compare.'
            }
        
        try:
            submissions = []
            for path in submission_paths:
                if os.path.exists(path):
                    sub = pd.read_csv(path)
                    sub_name = os.path.basename(path)
                    submissions.append((sub_name, sub))
                else:
                    return {
                        'error': f'Submission file not found: {path}'
                    }
            
            # Check if all submissions have PassengerId and Survived columns
            for name, sub in submissions:
                if 'PassengerId' not in sub.columns or 'Survived' not in sub.columns:
                    return {
                        'error': f'Submission {name} is missing required columns.'
                    }
            
            # Create a combined dataframe for comparison
            comparison_df = submissions[0][1][['PassengerId', 'Survived']].copy()
            comparison_df.columns = ['PassengerId', f'Survived_{submissions[0][0]}']
            
            for i in range(1, len(submissions)):
                name, sub = submissions[i]
                comparison_df = comparison_df.merge(
                    sub[['PassengerId', 'Survived']].rename(columns={'Survived': f'Survived_{name}'}),
                    on='PassengerId',
                    how='outer'
                )
            
            # Find differences between submissions
            survived_cols = [col for col in comparison_df.columns if col.startswith('Survived_')]
            
            # Calculate agreement percentage
            agreement_count = 0
            for idx, row in comparison_df.iterrows():
                values = [row[col] for col in survived_cols]
                if len(set(values)) == 1:  # All models agree
                    agreement_count += 1
            
            agreement_pct = (agreement_count / len(comparison_df)) * 100
            
            # Find passengers with different predictions
            differences = []
            for idx, row in comparison_df.iterrows():
                values = [row[col] for col in survived_cols]
                if len(set(values)) > 1:  # Models disagree
                    passenger_id = row['PassengerId']
                    predictions = {col.replace('Survived_', ''): int(row[col]) for col in survived_cols}
                    differences.append({
                        'PassengerId': passenger_id,
                        'Predictions': predictions
                    })
            
            # Generate comparison statistics
            stats = {}
            for name, sub in submissions:
                survived_count = sub['Survived'].sum()
                total_count = len(sub)
                survival_rate = (survived_count / total_count) * 100
                stats[name] = {
                    'survived_count': int(survived_count),
                    'total_count': total_count,
                    'survival_rate': survival_rate
                }
            
            return {
                'agreement_percentage': agreement_pct,
                'different_predictions_count': len(differences),
                'differences': differences[:20],  # Limit to first 20 differences
                'statistics': stats
            }
        except Exception as e:
            return {
                'error': f'Error comparing submissions: {str(e)}'
            }
    
    def export_submission(self, model_name, submission_df, output_format='csv'):
        """
        Export a submission to a file with specified format.
        
        Args:
            model_name (str): Name of the model used for predictions.
            submission_df (pandas.DataFrame): Submission dataframe with predictions.
            output_format (str, optional): Output format (csv or json).
            
        Returns:
            str: Path to the exported file.
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create base file name
            base_file_name = f"submission_{model_name}_{timestamp}"
            
            if output_format.lower() == 'csv':
                file_name = f"{base_file_name}.csv"
                file_path = os.path.join(self.full_submissions_dir, file_name)
                submission_df.to_csv(file_path, index=False)
            elif output_format.lower() == 'json':
                file_name = f"{base_file_name}.json"
                file_path = os.path.join(self.full_submissions_dir, file_name)
                submission_df.to_json(file_path, orient='records')
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Submission exported to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error exporting submission: {e}")
            raise
    
    def list_submissions(self):
        """
        List all submission files.
        
        Returns:
            list: List of submission file paths.
        """
        submissions = []
        
        if os.path.exists(self.full_submissions_dir):
            for file in os.listdir(self.full_submissions_dir):
                if file.endswith('.csv'):
                    submissions.append(os.path.join(self.full_submissions_dir, file))
        
        return submissions
    
    def get_submission_details(self, submission_path):
        """
        Get detailed information about a submission file.
        
        Args:
            submission_path (str): Path to the submission file.
            
        Returns:
            dict: Details about the submission.
        """
        try:
            if not os.path.exists(submission_path):
                return {'error': 'Submission file not found.'}
            
            # Get file metadata
            file_name = os.path.basename(submission_path)
            file_size = os.path.getsize(submission_path)
            file_modified = datetime.fromtimestamp(os.path.getmtime(submission_path))
            
            # Parse the submission file
            submission = pd.read_csv(submission_path)
            
            # Calculate statistics
            total_passengers = len(submission)
            survived_count = submission['Survived'].sum() if 'Survived' in submission.columns else 0
            survival_rate = (survived_count / total_passengers) * 100 if total_passengers > 0 else 0
            
            # Extract model name from file name if possible
            model_name = 'Unknown'
            if '_' in file_name:
                parts = file_name.split('_')
                if len(parts) > 1:
                    model_name = parts[1]
            
            return {
                'file_name': file_name,
                'file_size': file_size,
                'file_modified': file_modified.strftime('%Y-%m-%d %H:%M:%S'),
                'total_passengers': total_passengers,
                'survived_count': int(survived_count),
                'survival_rate': survival_rate,
                'model_name': model_name,
                'is_valid': self.validate_submission(submission_path)['valid']
            }
        except Exception as e:
            return {'error': f'Error getting submission details: {str(e)}'}
