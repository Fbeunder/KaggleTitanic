# -*- coding: utf-8 -*-
"""
Submission Generator Module

This module generates submission files for the Kaggle Titanic competition.
"""

import os
import pandas as pd

from src.data_processing.data_loader import DataLoader
from src.utilities.utils import create_directory_if_not_exists, get_project_root
from src.utilities.config import get_config


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
        
        print(f"Submission saved to {submission_path}")
        return submission_path
    
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
