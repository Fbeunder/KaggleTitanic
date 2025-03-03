"""
Fixtures en configuratie voor pytest.
Dit bestand bevat gedeelde fixtures die in meerdere testmodules gebruikt kunnen worden.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Zorg ervoor dat de src directory in de Python path staat
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importeer de benodigde modules
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.feature_engineering.feature_creator import FeatureCreator
from src.feature_engineering.feature_selector import FeatureSelector
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer
from src.modelling.model_evaluator import ModelEvaluator
from src.utilities.submission_generator import SubmissionGenerator
from src.web_interface.app import app as flask_app


@pytest.fixture
def sample_data():
    """Maak een klein sample van testdata."""
    data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Survived': [0, 1, 1, 0, 1],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 
                'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
                'Heikkinen, Miss. Laina',
                'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
                'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05],
        'Cabin': [None, 'C85', None, 'C123', None],
        'Embarked': ['S', 'C', 'S', 'S', 'S']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Maak een klein sample van test data (zonder Survived kolom)."""
    data = {
        'PassengerId': [892, 893, 894, 895, 896],
        'Pclass': [3, 3, 2, 3, 3],
        'Name': ['Kelly, Mr. James', 
                'Wilkes, Mrs. James (Ellen Needs)',
                'Myles, Mr. Thomas Francis',
                'Wirz, Mr. Albert',
                'Hirvonen, Mrs. Alexander (Helga E Lindqvist)'],
        'Sex': ['male', 'female', 'male', 'male', 'female'],
        'Age': [34.5, 47.0, 62.0, 27.0, 22.0],
        'SibSp': [0, 1, 0, 0, 1],
        'Parch': [0, 0, 0, 0, 1],
        'Ticket': ['330911', '363272', '240276', '315154', '3101298'],
        'Fare': [7.8292, 7.0, 9.6875, 8.6625, 12.2875],
        'Cabin': [None, None, None, None, None],
        'Embarked': ['Q', 'S', 'Q', 'S', 'S']
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_loader():
    """Fixture voor het maken van een DataLoader object."""
    return DataLoader()


@pytest.fixture
def preprocessed_data(sample_data, sample_test_data):
    """Fixture voor het maken van voorbewerkte data."""
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test = preprocessor.preprocess(sample_data, sample_test_data)
    return X_train, y_train, X_test


@pytest.fixture
def feature_creator():
    """Fixture voor het maken van een FeatureCreator object."""
    return FeatureCreator()


@pytest.fixture
def feature_selector():
    """Fixture voor het maken van een FeatureSelector object."""
    return FeatureSelector()


@pytest.fixture
def model_factory():
    """Fixture voor het maken van een ModelFactory object."""
    return ModelFactory()


@pytest.fixture
def model_trainer():
    """Fixture voor het maken van een ModelTrainer object."""
    return ModelTrainer()


@pytest.fixture
def model_evaluator():
    """Fixture voor het maken van een ModelEvaluator object."""
    return ModelEvaluator()


@pytest.fixture
def submission_generator():
    """Fixture voor het maken van een SubmissionGenerator object."""
    return SubmissionGenerator()


@pytest.fixture
def trained_model(preprocessed_data, model_factory, model_trainer):
    """Fixture voor het maken van een getraind model."""
    X_train, y_train, _ = preprocessed_data
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    model = model_factory.create_model('logistic_regression')
    model_trainer.train(model, X_train_sub, y_train_sub)
    return model, X_val, y_val


@pytest.fixture
def flask_test_client():
    """Fixture voor het maken van een Flask test client."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client
