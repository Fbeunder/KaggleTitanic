# Model Factory Documentation

## Overview

The Model Factory module provides a standardized way to create, train, and evaluate different machine learning models for the Titanic survival prediction problem. It implements the factory pattern to allow easy instantiation of different model types while maintaining a consistent interface.

## Model Hierarchy

- `TitanicModel`: Abstract base class defining the common interface for all models
- Concrete implementations:
  - `LogisticRegressionModel`: Logistic Regression model
  - `RandomForestModel`: Random Forest Classifier model
  - `DecisionTreeModel`: Decision Tree Classifier model
  - `SVMModel`: Support Vector Machine model
  - `KNeighborsModel`: K-Nearest Neighbors model
  - `GradientBoostingModel`: Gradient Boosting Classifier model
- `ModelFactory`: Factory class that creates and configures model instances

## Common Model Interface

All models implement a common interface:

- `fit(X, y)`: Train the model on training data
- `predict(X)`: Make predictions on new data
- `predict_proba(X)`: Get probability estimates for predictions
- `evaluate(X, y)`: Evaluate model performance on test data
- `tune_hyperparameters(X, y, param_grid, cv, scoring)`: Tune hyperparameters using GridSearchCV
- `get_feature_importance(feature_names)`: Get feature importance if supported by the model

## Using the Model Factory

### Creating a Model

```python
from src.modelling.model_factory import ModelFactory

# Create a default model
rf_model = ModelFactory.create_model('random_forest')

# Create a model with custom parameters
custom_rf = ModelFactory.create_model(
    'random_forest',
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2
)

# Get available model types
available_models = ModelFactory.get_available_models()
print(f"Available models: {available_models}")

# Get default parameters for a model type
default_params = ModelFactory.get_model_default_params('logistic_regression')
print(f"Default parameters: {default_params}")
```

### Training and Evaluating a Model

```python
# Train a model
model = ModelFactory.create_model('random_forest')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get probability of positive class

# Evaluate the model
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC AUC: {metrics['roc_auc']:.4f}")

# Get feature importance
importance_df = model.get_feature_importance(feature_names=X_train.columns)
if importance_df is not None:
    print(importance_df.head(10))
```

### Hyperparameter Tuning

```python
# Tune hyperparameters with default parameter grid
model = ModelFactory.create_model('svm')
model.tune_hyperparameters(X_train, y_train, cv=5, scoring='accuracy')

# Tune with custom parameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf']
}
model.tune_hyperparameters(X_train, y_train, param_grid=param_grid)
```

## Adding New Models

To add a new model type to the factory, you need to:

1. Create a new class that inherits from `TitanicModel`
2. Implement the required methods
3. Register the model with the factory

### Example: Adding a New Model

```python
from sklearn.ensemble import AdaBoostClassifier
from src.modelling.model_factory import TitanicModel, ModelFactory

class AdaBoostModel(TitanicModel):
    """
    AdaBoost model implementation for Titanic survival prediction.
    """
    
    def get_default_params(self):
        """Get default parameters for AdaBoost."""
        return {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R',
            'random_state': 42
        }
    
    def _initialize_model(self):
        """Initialize the AdaBoost model."""
        self.model = AdaBoostClassifier(**self.params)
    
    def get_param_grid(self):
        """Get parameter grid for hyperparameter tuning."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }

# Register the new model type with the factory
ModelFactory.register_model('adaboost', AdaBoostModel)

# Now you can create AdaBoost models
ada_model = ModelFactory.create_model('adaboost')
```

## Model Configuration Options

Each model type has its own set of configurable parameters. You can view the default parameters for each model type using:

```python
ModelFactory.get_model_default_params('model_type')
```

Where `model_type` is one of the available model types (e.g., 'random_forest', 'logistic_regression', etc.).
