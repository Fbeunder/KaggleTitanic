# -*- coding: utf-8 -*-
"""
Main Web Application Module

This module is the entry point for the web application interface of the Titanic survival prediction project.
It uses Flask to create a web application that allows users to interact with the machine learning models.
"""

import os
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file

from src.web_interface.model_interface import ModelInterface
from src.data_processing.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize model interface
model_interface = ModelInterface()

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-titanic-app')
    
    # Load data at startup to avoid loading it for each request
    try:
        model_interface.load_data()
        logger.info("Data loaded successfully at app startup")
    except Exception as e:
        logger.error(f"Error loading data at startup: {e}")
    
    # Register routes
    @app.route('/')
    def index():
        """Home page route."""
        return render_template('index.html')
    
    @app.route('/features')
    def features():
        """Feature exploration page route."""
        try:
            # Get dataset statistics for the template
            data_loader = DataLoader()
            train_data = data_loader.load_train_data()
            
            if train_data is None:
                flash("Error loading training data. Please check data files.", "error")
                return render_template('features.html', dataset_stats={})
            
            # Calculate basic statistics
            total_passengers = len(train_data)
            survival_count = train_data['Survived'].sum()
            survival_rate = (survival_count / total_passengers) * 100
            
            avg_age = train_data['Age'].mean()
            avg_fare = train_data['Fare'].mean()
            
            male_data = train_data[train_data['Sex'] == 'male']
            female_data = train_data[train_data['Sex'] == 'female']
            male_count = len(male_data)
            female_count = len(female_data)
            male_pct = (male_count / total_passengers) * 100
            female_pct = (female_count / total_passengers) * 100
            
            missing_age = train_data['Age'].isnull().sum()
            missing_age_pct = (missing_age / total_passengers) * 100
            
            # Create statistics dictionary
            dataset_stats = {
                'total_passengers': total_passengers,
                'survival_count': survival_count,
                'survival_rate': survival_rate,
                'avg_age': avg_age,
                'avg_fare': avg_fare,
                'male_count': male_count,
                'female_count': female_count,
                'male_pct': male_pct,
                'female_pct': female_pct,
                'missing_age': missing_age,
                'missing_age_pct': missing_age_pct
            }
            
            return render_template('features.html', dataset_stats=dataset_stats)
        except Exception as e:
            logger.error(f"Error in features route: {e}")
            flash(f"An error occurred while loading feature data: {str(e)}", "error")
            return render_template('features.html', dataset_stats={})
    
    @app.route('/train', methods=['GET', 'POST'])
    def train():
        """Model training page route."""
        if request.method == 'POST':
            try:
                # Get form data
                model_type = request.form.get('model_type')
                feature_engineering = request.form.get('feature_engineering')
                cross_validation = int(request.form.get('cross_validation', 5))
                test_size = float(request.form.get('test_size', 0.2)) / 100  # Convert percentage to proportion
                hyperparameter_tuning = bool(request.form.get('hyperparameter_tuning', False))
                
                # Additional parameters would be processed here depending on the model type
                
                # Train the model
                feature_eng_enabled = feature_engineering in ['enhanced', 'all']
                success = model_interface.train_model(
                    model_name=model_type,
                    hyperparameter_tuning=hyperparameter_tuning,
                    feature_engineering=feature_eng_enabled
                )
                
                if success:
                    flash(f"Successfully trained {model_type} model!", "success")
                    return redirect(url_for('results'))
                else:
                    flash(f"Error training {model_type} model. Please check logs.", "error")
            except Exception as e:
                logger.error(f"Error in train POST route: {e}")
                flash(f"An error occurred during model training: {str(e)}", "error")
        
        return render_template('train.html')
    
    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        """Prediction page route."""
        if request.method == 'POST':
            try:
                # Get form data for a single passenger
                passenger_data = {
                    'Name': request.form.get('passenger_name', ''),
                    'Sex': request.form.get('passenger_sex', 'male'),
                    'Age': float(request.form.get('passenger_age', 0)) if request.form.get('passenger_age') else None,
                    'Pclass': int(request.form.get('passenger_pclass', 3)),
                    'Fare': float(request.form.get('passenger_fare', 0)) if request.form.get('passenger_fare') else None,
                    'SibSp': int(request.form.get('passenger_sibsp', 0)),
                    'Parch': int(request.form.get('passenger_parch', 0)),
                    'Embarked': request.form.get('passenger_embarked', 'S'),
                    'Cabin': request.form.get('passenger_cabin', '')
                }
                
                # Get selected model
                model_name = request.form.get('model_select', 'random_forest')
                
                # Make prediction
                # In a real application, this would use the model_interface to predict
                # For now, return mock data
                prediction_result = {
                    'survived': 1,  # 1 for survived, 0 for not survived
                    'probability': 0.82,  # Probability of survival
                    'key_factors': [
                        {'factor': 'Gender: Female', 'impact': 30},
                        {'factor': 'Passenger Class: 1st', 'impact': 25},
                        {'factor': 'Age: 24', 'impact': 15},
                        {'factor': 'Family Size: 0', 'impact': -10}
                    ],
                    'similar_passengers': [
                        {'name': 'Brown, Mrs. James Joseph', 'age': 25, 'sex': 'Female', 
                         'pclass': '1st', 'fare': 63.50, 'embarked': 'C', 'survived': 1},
                        {'name': 'Graham, Miss. Margaret Edith', 'age': 19, 'sex': 'Female', 
                         'pclass': '1st', 'fare': 30.00, 'embarked': 'S', 'survived': 1},
                        {'name': 'Harper, Mrs. Henry Sleeper', 'age': 49, 'sex': 'Female', 
                         'pclass': '1st', 'fare': 76.73, 'embarked': 'C', 'survived': 1}
                    ]
                }
                
                return jsonify(prediction_result)
            except Exception as e:
                logger.error(f"Error in predict POST route: {e}")
                return jsonify({'error': str(e)}), 400
        
        return render_template('predict.html')
    
    @app.route('/results')
    def results():
        """Model results page route."""
        # In a real application, this would retrieve actual model results from the model_interface
        return render_template('results.html')
    
    @app.route('/api/feature-importance')
    def feature_importance():
        """API endpoint for feature importance data."""
        try:
            model_name = request.args.get('model', 'random_forest')
            # In a real application, this would get actual feature importance from the model
            # For now, return mock data
            importance_data = {
                'features': ['Sex', 'Pclass', 'Fare', 'Age', 'Embarked', 'FamilySize', 'Title', 'CabinDeck'],
                'importance': [0.35, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]
            }
            return jsonify(importance_data)
        except Exception as e:
            logger.error(f"Error in feature-importance API: {e}")
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/model-comparison')
    def model_comparison():
        """API endpoint for model comparison data."""
        try:
            # In a real application, this would get actual model comparison data
            # For now, return mock data
            comparison_data = {
                'models': ['Gradient Boosting', 'Random Forest', 'SVM', 'Logistic Regression', 'KNN', 'Decision Tree'],
                'accuracy': [85.6, 84.2, 83.1, 82.5, 80.7, 78.9],
                'precision': [83.4, 82.1, 81.3, 79.3, 78.9, 75.6],
                'recall': [84.1, 82.8, 80.5, 83.7, 77.2, 79.3],
                'f1': [83.7, 82.4, 80.9, 81.4, 78.0, 77.4],
                'auc': [0.891, 0.877, 0.854, 0.857, 0.821, 0.794]
            }
            return jsonify(comparison_data)
        except Exception as e:
            logger.error(f"Error in model-comparison API: {e}")
            return jsonify({'error': str(e)}), 400
            
    @app.route('/api/generate-submission', methods=['POST'])
    def generate_submission():
        """API endpoint for generating Kaggle submission files."""
        try:
            # Get model name from request
            model_name = request.form.get('model_name')
            file_format = request.form.get('format', 'csv')
            custom_file_name = request.form.get('file_name')
            
            # Generate the submission file
            result = model_interface.generate_kaggle_submission(
                model_name=model_name,
                custom_file_name=custom_file_name,
                format=file_format
            )
            
            if not result['success']:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Unknown error generating submission')
                }), 400
            
            return jsonify({
                'success': True,
                'file_path': result['file_path'],
                'model_name': result['model_name'],
                'validation': result['validation']
            })
        except Exception as e:
            logger.error(f"Error in generate-submission API: {e}")
            return jsonify({'success': False, 'error': str(e)}), 400
    
    @app.route('/api/list-submissions')
    def list_submissions():
        """API endpoint for listing available submission files."""
        try:
            submissions = model_interface.list_available_submissions()
            return jsonify({'submissions': submissions})
        except Exception as e:
            logger.error(f"Error in list-submissions API: {e}")
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/download-submission')
    def download_submission():
        """API endpoint for downloading a submission file."""
        try:
            file_path = request.args.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            return send_file(file_path, as_attachment=True)
        except Exception as e:
            logger.error(f"Error in download-submission API: {e}")
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/compare-submissions', methods=['POST'])
    def compare_submissions():
        """API endpoint for comparing multiple submission files."""
        try:
            # Get submission file paths from request
            submission_paths = request.json.get('submission_paths', [])
            if not submission_paths or len(submission_paths) < 2:
                return jsonify({'error': 'Need at least two submission files to compare'}), 400
            
            # Check if all files exist
            missing_files = [path for path in submission_paths if not os.path.exists(path)]
            if missing_files:
                return jsonify({
                    'error': f'The following submission files were not found: {missing_files}'
                }), 404
            
            # Use submission generator to compare the files
            comparison_result = model_interface.submission_generator.compare_submissions(submission_paths)
            return jsonify(comparison_result)
        except Exception as e:
            logger.error(f"Error in compare-submissions API: {e}")
            return jsonify({'error': str(e)}), 400
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {e}")
        return render_template('500.html'), 500
    
    return app

# Create the Flask app
app = create_app()

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
