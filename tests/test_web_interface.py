"""
Tests voor de web interface (app.py, model_interface.py).
"""
import pytest
import os
import json
import pandas as pd
from flask import url_for

# Markeer deze tests als web tests en integratie tests
pytestmark = [pytest.mark.webtest, pytest.mark.integration]


class TestWebRoutes:
    """
    Tests voor de routes van de Flask applicatie.
    """
    
    def test_home_page(self, flask_test_client):
        """Test of de homepage correct laadt."""
        response = flask_test_client.get('/')
        assert response.status_code == 200
        assert b'Titanic Survival Predictor' in response.data
    
    def test_features_page(self, flask_test_client):
        """Test of de features pagina correct laadt."""
        response = flask_test_client.get('/features')
        assert response.status_code == 200
        assert b'Feature Engineering' in response.data
    
    def test_train_page(self, flask_test_client):
        """Test of de train pagina correct laadt."""
        response = flask_test_client.get('/train')
        assert response.status_code == 200
        assert b'Model Training' in response.data
    
    def test_predict_page(self, flask_test_client):
        """Test of de predict pagina correct laadt."""
        response = flask_test_client.get('/predict')
        assert response.status_code == 200
        assert b'Prediction' in response.data
    
    def test_results_page(self, flask_test_client):
        """Test of de results pagina correct laadt."""
        response = flask_test_client.get('/results')
        assert response.status_code == 200
        assert b'Model Results' in response.data
    
    def test_api_models(self, flask_test_client):
        """Test de API endpoint voor model info."""
        response = flask_test_client.get('/api/models')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        # Controleer of de belangrijkste modellen in de lijst staan
        model_names = [model['name'] for model in data]
        assert 'logistic_regression' in model_names
        assert 'random_forest' in model_names
    
    def test_nonexistent_page(self, flask_test_client):
        """Test of een niet-bestaande pagina een 404 error geeft."""
        response = flask_test_client.get('/pagina-bestaat-niet')
        assert response.status_code == 404


class TestModelInterface:
    """
    Tests voor de model_interface.py module.
    """
    
    def test_predict_passenger(self, flask_test_client):
        """Test het voorspellen van een passagier via de API."""
        # Testdata voor een passagier
        passenger_data = {
            'pclass': 1,
            'sex': 'female',
            'age': 29,
            'sibsp': 0,
            'parch': 0,
            'fare': 211.3375,
            'embarked': 'S'
        }
        
        # Stuur het verzoek naar de API
        response = flask_test_client.post(
            '/api/predict',
            data=json.dumps(passenger_data),
            content_type='application/json'
        )
        
        # Controleer of we een geldige respons krijgen
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'survival_probability' in data
        assert isinstance(data['survival_probability'], float)
        assert 0 <= data['survival_probability'] <= 1
    
    def test_train_model(self, flask_test_client):
        """Test het trainen van een model via de API."""
        # Configuratie voor het trainen van een model
        train_config = {
            'model_type': 'logistic_regression',
            'hyperparameters': {
                'C': 1.0,
                'max_iter': 100
            },
            'feature_selection': False
        }
        
        # Stuur het verzoek naar de API
        response = flask_test_client.post(
            '/api/train',
            data=json.dumps(train_config),
            content_type='application/json'
        )
        
        # Controleer of we een geldige respons krijgen
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'model_id' in data
        assert 'accuracy' in data
        assert isinstance(data['accuracy'], float)
        assert 0 <= data['accuracy'] <= 1


class TestEndToEnd:
    """
    End-to-end tests voor de web applicatie.
    """
    
    def test_full_prediction_workflow(self, flask_test_client):
        """Test de volledige workflow van het laden van een model en het maken van een voorspelling."""
        # Stap 1: Train een model
        train_config = {
            'model_type': 'logistic_regression',
            'hyperparameters': {
                'C': 1.0,
                'max_iter': 100
            }
        }
        
        response = flask_test_client.post(
            '/api/train',
            data=json.dumps(train_config),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        train_data = json.loads(response.data)
        model_id = train_data['model_id']
        
        # Stap 2: Gebruik het model om een voorspelling te maken
        passenger_data = {
            'pclass': 1,
            'sex': 'female',
            'age': 29,
            'sibsp': 0,
            'parch': 0,
            'fare': 211.3375,
            'embarked': 'S',
            'model_id': model_id  # Gebruik het model dat we net hebben getraind
        }
        
        response = flask_test_client.post(
            '/api/predict',
            data=json.dumps(passenger_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        prediction_data = json.loads(response.data)
        assert 'survival_probability' in prediction_data
        assert isinstance(prediction_data['survival_probability'], float)
        
        # Stap 3: Genereer een submission file
        response = flask_test_client.post(
            '/api/generate_submission',
            data=json.dumps({'model_id': model_id}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        submission_data = json.loads(response.data)
        assert 'submission_id' in submission_data
        assert 'download_url' in submission_data
