# -*- coding: utf-8 -*-
"""
Submission Generator Module

This module generates submission files for the Kaggle Titanic competition.
It provides functionality to create, validate, compare, and export prediction files.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
    for the Kaggle Titanic competition, validates them, and provides
    utilities for comparing different model submissions.
    """
    
    def __init__(self):
        """
        Initialize the SubmissionGenerator.
        """
        self.config = get_config()
        self.submissions_dir = self.config.get('submissions_dir', 'submissions')
        self.full_submissions_dir = os.path.join(get_project_root(), self.submissions_dir)
        create_directory_if_not_exists(self.full_submissions_dir)
        self.last_submission = None
    
    def generate_submission(self, model, X_test, file_name=None, description=None):
        """
        Generate a submission file for the Kaggle competition.
        
        Args:
            model: Trained model to use for predictions.
            X_test (pandas.DataFrame): Test data features.
            file_name (str, optional): Name of the submission file.
                                     If None, a default name is generated.
            description (str, optional): Description of the submission for tracking.
                                     
        Returns:
            str: Path to the generated submission file.
        """
        # Make predictions
        logger.info(f"Generating predictions using model: {model.__class__.__name__}")
        predictions = model.predict(X_test)
        
        # Load the original test data to get passenger IDs
        loader = DataLoader()
        test_data = loader.load_test_data()
        
        if test_data is None:
            raise ValueError("Could not load test data.")
        
        # Convert predictions to integers (0 or 1)
        predictions = np.round(predictions).astype(int)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': test_data['PassengerId'],
            'Survived': predictions
        })
        
        # Generate file name if not provided
        if file_name is None:
            model_name = model.__class__.__name__
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"submission_{model_name}_{timestamp}.csv"
        
        # Make sure file has .csv extension
        if not file_name.endswith('.csv'):
            file_name += '.csv'
        
        # Save submission
        submission_path = os.path.join(self.full_submissions_dir, file_name)
        submission.to_csv(submission_path, index=False)
        
        # Add metadata file with description and statistics
        if description:
            metadata_path = os.path.join(
                self.full_submissions_dir, 
                os.path.splitext(file_name)[0] + '_metadata.txt'
            )
            
            # Calculate basic statistics
            survived_count = predictions.sum()
            total_count = len(predictions)
            survival_rate = survived_count / total_count * 100
            
            with open(metadata_path, 'w') as f:
                f.write(f"Submission: {file_name}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {model.__class__.__name__}\n")
                f.write(f"Description: {description}\n\n")
                f.write(f"Statistics:\n")
                f.write(f"  Total passengers: {total_count}\n")
                f.write(f"  Predicted survivors: {survived_count} ({survival_rate:.2f}%)\n")
                f.write(f"  Predicted casualties: {total_count - survived_count} ({100 - survival_rate:.2f}%)\n")
        
        # Store last submission for quick reference
        self.last_submission = {
            'path': submission_path,
            'data': submission,
            'model_name': model.__class__.__name__,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Submission saved to {submission_path}")
        return submission_path
    
    def validate_submission(self, submission_path=None, submission_df=None):
        """
        Validate a submission file for the Kaggle competition.
        
        Args:
            submission_path (str, optional): Path to the submission file.
            submission_df (pandas.DataFrame, optional): DataFrame containing the submission.
            
        Returns:
            dict: Validation results with any issues found.
        """
        # Get submission data
        if submission_df is None and submission_path is None:
            if self.last_submission:
                submission_df = self.last_submission['data']
            else:
                raise ValueError("No submission provided and no last submission available.")
        
        if submission_df is None:
            try:
                submission_df = pd.read_csv(submission_path)
            except Exception as e:
                return {'valid': False, 'errors': [f"Failed to read submission file: {str(e)}"]}
        
        validation_results = {'valid': True, 'warnings': [], 'errors': []}
        
        # Check required columns
        required_columns = ['PassengerId', 'Survived']
        missing_columns = [col for col in required_columns if col not in submission_df.columns]
        
        if missing_columns:
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
            return validation_results
        
        # Load test data to check all passengers are included
        try:
            loader = DataLoader()
            test_data = loader.load_test_data()
            test_passenger_ids = set(test_data['PassengerId'])
            submission_passenger_ids = set(submission_df['PassengerId'])
            
            # Check if all passenger IDs are included
            missing_ids = test_passenger_ids - submission_passenger_ids
            if missing_ids:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Missing {len(missing_ids)} PassengerId(s) in submission"
                )
            
            # Check if there are extra passenger IDs
            extra_ids = submission_passenger_ids - test_passenger_ids
            if extra_ids:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Found {len(extra_ids)} extra PassengerId(s) in submission"
                )
        except Exception as e:
            validation_results['warnings'].append(
                f"Could not verify passenger IDs against test data: {str(e)}"
            )
        
        # Check Survived values
        valid_survived_values = {0, 1}
        invalid_survived = [
            value for value in submission_df['Survived'].unique() 
            if value not in valid_survived_values
        ]
        
        if invalid_survived:
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"Invalid Survived values found: {invalid_survived}. Only 0 and 1 are allowed."
            )
        
        # Check for NaN values
        nan_count = submission_df.isna().sum().sum()
        if nan_count > 0:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Found {nan_count} NaN values in submission")
        
        # Check survival rate for suspicious values
        survival_rate = submission_df['Survived'].mean() * 100
        if survival_rate < 15 or survival_rate > 50:
            validation_results['warnings'].append(
                f"Unusual survival rate: {survival_rate:.2f}%. Expected range is 15-50%."
            )
        
        return validation_results
    
    def compare_submissions(self, submission_paths=None, names=None):
        """
        Compare multiple submission files to identify differences.
        
        Args:
            submission_paths (list, optional): List of paths to submission files.
                                            If None, compares all files in the submissions directory.
            names (list, optional): Custom names for the submissions.
            
        Returns:
            pandas.DataFrame: Comparison results.
        """
        # Get submission files
        if submission_paths is None:
            submission_paths = [
                os.path.join(self.full_submissions_dir, f) 
                for f in os.listdir(self.full_submissions_dir) 
                if f.endswith('.csv')
            ]
        
        if not submission_paths:
            logger.warning("No submission files found for comparison")
            return pd.DataFrame()
        
        # Load submissions
        submissions = []
        for i, path in enumerate(submission_paths):
            try:
                df = pd.read_csv(path)
                name = names[i] if names and i < len(names) else os.path.basename(path)
                submissions.append((name, df))
            except Exception as e:
                logger.error(f"Failed to read submission file {path}: {e}")
        
        if len(submissions) < 2:
            logger.warning("Need at least two valid submissions for comparison")
            return pd.DataFrame()
        
        # Create combined DataFrame
        combined_df = submissions[0][1][['PassengerId']].copy()
        
        for name, df in submissions:
            combined_df[name] = df['Survived']
        
        # Create comparison statistics
        stats = []
        
        # Pairwise agreement
        for i in range(len(submissions)):
            for j in range(i+1, len(submissions)):
                name_i = submissions[i][0]
                name_j = submissions[j][0]
                
                agreement = (combined_df[name_i] == combined_df[name_j]).mean() * 100
                
                stats.append({
                    'Comparison': f"{name_i} vs {name_j}",
                    'Agreement (%)': agreement,
                    'Differences': (combined_df[name_i] != combined_df[name_j]).sum()
                })
        
        # Individual statistics
        for name, df in submissions:
            survival_rate = df['Survived'].mean() * 100
            stats.append({
                'Comparison': f"{name} (survival rate)",
                'Agreement (%)': None,
                'Differences': None,
                'Survival Rate (%)': survival_rate,
                'Survivors': df['Survived'].sum(),
                'Non-survivors': (df['Survived'] == 0).sum()
            })
        
        # Find passengers with most disagreement
        disagreement_counts = pd.Series(0, index=combined_df['PassengerId'])
        
        for i in range(1, len(submissions)):
            name_i = submissions[i][0]
            for j in range(i+1, len(submissions)):
                name_j = submissions[j][0]
                
                # Find rows where predictions differ
                diff_mask = combined_df[name_i] != combined_df[name_j]
                disagreement_counts = disagreement_counts.add(
                    combined_df.loc[diff_mask, 'PassengerId'].value_counts(),
                    fill_value=0
                )
        
        # Get top disagreed passengers
        top_disagreed = disagreement_counts.sort_values(ascending=False).head(10)
        
        if not top_disagreed.empty:
            logger.info("Top passengers with prediction disagreements:")
            for passenger_id, count in top_disagreed.items():
                predictions = {
                    name: df.loc[df['PassengerId'] == passenger_id, 'Survived'].values[0]
                    for name, df in submissions
                }
                logger.info(f"PassengerId {passenger_id}: {count} disagreements - {predictions}")
        
        return pd.DataFrame(stats)
    
    def export_submission(self, submission_path=None, output_path=None):
        """
        Export a submission file to a specified location.
        
        Args:
            submission_path (str, optional): Path to the submission file.
                                         If None, uses the last generated submission.
            output_path (str, optional): Path where to export the submission.
                                      If None, exports to the current directory.
            
        Returns:
            str: Path to the exported file.
        """
        # Get submission path
        if submission_path is None:
            if self.last_submission:
                submission_path = self.last_submission['path']
            else:
                raise ValueError("No submission path provided and no last submission available.")
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(os.getcwd(), os.path.basename(submission_path))
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Copy the file
        try:
            # Read the submission
            submission_df = pd.read_csv(submission_path)
            
            # Validate before exporting
            validation_results = self.validate_submission(submission_df=submission_df)
            if not validation_results['valid']:
                logger.warning(f"Exporting invalid submission: {validation_results['errors']}")
            
            # Save to output path
            submission_df.to_csv(output_path, index=False)
            logger.info(f"Submission exported to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Failed to export submission: {e}")
            raise
    
    def plot_prediction_distribution(self, submission_path=None, ax=None, figsize=(10, 6)):
        """
        Plot the distribution of survival predictions by passenger class, sex, etc.
        
        Args:
            submission_path (str, optional): Path to the submission file.
                                         If None, uses the last generated submission.
            ax (matplotlib.axes.Axes, optional): Axes to plot on.
            figsize (tuple, optional): Figure size.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Get submission data
        if submission_path is None:
            if self.last_submission:
                submission_path = self.last_submission['path']
            else:
                raise ValueError("No submission path provided and no last submission available.")
        
        try:
            # Load submission and test data
            submission_df = pd.read_csv(submission_path)
            
            loader = DataLoader()
            test_data = loader.load_test_data()
            
            # Merge submission with test data
            merged_data = pd.merge(
                submission_df, 
                test_data,
                on='PassengerId'
            )
            
            # Create plot
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            
            # Calculate survival rates by passenger class and sex
            survival_by_class_sex = merged_data.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
            
            # Plot
            survival_by_class_sex.plot(kind='bar', ax=ax)
            
            ax.set_xlabel('Passenger Class')
            ax.set_ylabel('Survival Rate')
            ax.set_title('Predicted Survival Rate by Passenger Class and Sex')
            ax.legend(title='Sex')
            
            # Add text with overall survival rate
            overall_rate = merged_data['Survived'].mean() * 100
            ax.text(
                0.02, 0.95, 
                f'Overall Survival Rate: {overall_rate:.1f}%',
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5)
            )
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Failed to plot prediction distribution: {e}")
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                return fig
            return ax.figure
    
    def list_submissions(self):
        """
        List all submission files with metadata.
        
        Returns:
            pandas.DataFrame: Submission files with metadata.
        """
        submissions = []
        
        if os.path.exists(self.full_submissions_dir):
            for file in os.listdir(self.full_submissions_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.full_submissions_dir, file)
                    
                    # Get file info
                    stats = os.stat(file_path)
                    timestamp = datetime.fromtimestamp(stats.st_mtime)
                    
                    # Read submission to get basic statistics
                    try:
                        df = pd.read_csv(file_path)
                        survivor_count = df['Survived'].sum()
                        total_count = len(df)
                        survival_rate = survivor_count / total_count * 100
                    except:
                        survivor_count = None
                        total_count = None
                        survival_rate = None
                    
                    # Check if there's a metadata file
                    metadata_path = os.path.join(
                        self.full_submissions_dir, 
                        os.path.splitext(file)[0] + '_metadata.txt'
                    )
                    
                    description = None
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                content = f.read()
                                # Extract description
                                for line in content.split('\n'):
                                    if line.startswith('Description:'):
                                        description = line.replace('Description:', '').strip()
                                        break
                        except:
                            pass
                    
                    submissions.append({
                        'file_name': file,
                        'path': file_path,
                        'date': timestamp,
                        'size_kb': stats.st_size / 1024,
                        'survivor_count': survivor_count,
                        'total_count': total_count,
                        'survival_rate': survival_rate,
                        'description': description
                    })
        
        # Convert to DataFrame
        submissions_df = pd.DataFrame(submissions)
        
        # Sort by date (most recent first)
        if not submissions_df.empty:
            submissions_df = submissions_df.sort_values('date', ascending=False)
        
        return submissions_df
