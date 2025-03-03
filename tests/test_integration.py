"""
Integratie tests voor het testen van de samenwerking tussen verschillende modules.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Markeer dit als een integratie test
pytestmark = pytest.mark.integration


class TestDataToFeatures:
    """
    Test de integratie tussen data processing en feature engineering.
    """
    
    def test_preprocessing_to_feature_creation(self, sample_data, sample_test_data, 
                                             data_loader, feature_creator):
        """Test de workflow van data preprocessing naar feature creation."""
        # Bereid de data voor
        preprocessor = data_loader.preprocessor
        X_train, y_train, X_test = preprocessor.preprocess(sample_data, sample_test_data)
        
        # Maak features
        X_train_featured = feature_creator.create_all_features(X_train)
        X_test_featured = feature_creator.create_all_features(X_test)
        
        # Controleer of we de juiste resultaten krijgen
        assert isinstance(X_train_featured, pd.DataFrame)
        assert isinstance(X_test_featured, pd.DataFrame)
        assert len(X_train_featured) == len(X_train)
        assert len(X_test_featured) == len(X_test)
        
        # Controleer of nieuwe features zijn toegevoegd
        assert set(X_train_featured.columns) > set(X_train.columns)
        assert set(X_test_featured.columns) > set(X_test.columns)


class TestFeaturesToModel:
    """
    Test de integratie tussen feature engineering en modeling.
    """
    
    def test_feature_selection_to_modeling(self, preprocessed_data, feature_selector, 
                                          model_factory, model_trainer):
        """Test de workflow van feature selectie naar model training."""
        X_train, y_train, X_test = preprocessed_data
        
        # Voer feature selectie uit
        X_train_selected, X_test_selected, selected_features = feature_selector.select_k_best(
            X_train, y_train, X_test, k=3
        )
        
        # Train een model
        model = model_factory.create_model('logistic_regression')
        model_trainer.train(model, X_train_selected, y_train)
        
        # Maak voorspellingen
        predictions = model.predict(X_test_selected)
        
        # Controleer of we de juiste resultaten krijgen
        assert X_train_selected.shape[1] == 3  # We hebben 3 features geselecteerd
        assert X_test_selected.shape[1] == 3
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)  # Binaire voorspellingen


class TestModelToEvaluation:
    """
    Test de integratie tussen model training en evaluatie.
    """
    
    def test_model_training_to_evaluation(self, preprocessed_data, model_factory, 
                                         model_trainer, model_evaluator):
        """Test de workflow van model training naar evaluatie."""
        X_train, y_train, X_test = preprocessed_data
        
        # Split de data voor validatie
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        
        # Train een model
        model = model_factory.create_model('logistic_regression')
        model_trainer.train(model, X_train_sub, y_train_sub)
        
        # Evalueer het model
        metrics = model_evaluator.evaluate(model, X_val, y_val)
        
        # Controleer of de metrics juist zijn
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert all(0 <= metric <= 1 for metric in metrics.values())


class TestEndToEndPipeline:
    """
    Test de volledige pipeline van data tot submission generation.
    """
    
    def test_entire_pipeline(self, sample_data, sample_test_data, data_loader, 
                           feature_creator, feature_selector, model_factory, 
                           model_trainer, model_evaluator, submission_generator):
        """Test de volledige workflow van data tot submission generation."""
        # Stap 1: Preprocessing
        preprocessor = data_loader.preprocessor
        X_train, y_train, X_test = preprocessor.preprocess(sample_data, sample_test_data)
        
        # Stap 2: Feature engineering
        X_train_featured = feature_creator.create_all_features(X_train)
        X_test_featured = feature_creator.create_all_features(X_test)
        
        # Stap 3: Feature selectie
        X_train_selected, X_test_selected, selected_features = feature_selector.select_k_best(
            X_train_featured, y_train, X_test_featured, k=3
        )
        
        # Stap 4: Model training
        model = model_factory.create_model('random_forest')
        model_trainer.train(model, X_train_selected, y_train)
        
        # Stap 5: Model evaluatie
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_selected, y_train, test_size=0.2, random_state=42)
        metrics = model_evaluator.evaluate(model, X_val, y_val)
        
        # Stap 6: Submission generation
        predictions = model.predict(X_test_selected)
        submission = submission_generator.generate(sample_test_data['PassengerId'], predictions)
        
        # Controleer of de submission correct is
        assert isinstance(submission, pd.DataFrame)
        assert set(submission.columns) == {'PassengerId', 'Survived'}
        assert len(submission) == len(sample_test_data)
        assert all(pred in [0, 1] for pred in submission['Survived'])
