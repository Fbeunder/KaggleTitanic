# Getting Started with Titanic Survival Prediction

This guide will help you quickly get up and running with the Titanic Survival Prediction project. Follow these steps to set up your environment, understand the project structure, and start making predictions.

## Quick Start Guide

### Step 1: Set Up Your Environment

First, make sure you have Python 3.8 or higher installed on your system. Then:

1. Clone the repository:
   ```bash
   git clone https://github.com/Fbeunder/KaggleTitanic.git
   cd KaggleTitanic
   ```

2. Create and activate a virtual environment:
   ```bash
   # For macOS/Linux:
   python -m venv venv
   source venv/bin/activate
   
   # For Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you have the Titanic dataset files (`train.csv` and `test.csv`) in the root directory or the `data/` directory.

### Step 2: Explore the Data

The best way to get familiar with the dataset is to run the exploratory data analysis notebook:

```bash
jupyter notebook notebooks/titanic_eda.ipynb
```

This notebook will show you:
- The structure of the Titanic dataset
- Key statistics and distributions
- Interesting patterns and correlations
- Data quality issues and solutions

### Step 3: Run the Web Interface

The easiest way to interact with the project is through the web interface:

```bash
python -m src.web_interface.app
```

Open your browser and navigate to http://localhost:8050

Through the web interface, you can:
1. Explore the data with interactive visualizations
2. Configure and train different machine learning models
3. Adjust feature engineering settings
4. Make predictions and generate submission files

### Step 4: Use the API Programmatically

If you prefer to use the project as a Python library, here's a simple example:

```python
# Import key components
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.feature_engineering.feature_creator import FeatureCreator
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer
from src.modelling.model_evaluator import ModelEvaluator

# 1. Load data
loader = DataLoader()
train_data = loader.load_train_data()
test_data = loader.load_test_data()

# 2. Preprocess data
preprocessor = DataPreprocessor()
X_train, y_train = preprocessor.preprocess(train_data, is_training=True)
X_test = preprocessor.preprocess(test_data, is_training=False)

# 3. Create features
feature_creator = FeatureCreator()
X_train = feature_creator.create_all_features(X_train)
X_test = feature_creator.create_all_features(X_test)

# 4. Create and train a model
factory = ModelFactory()
model = factory.create_model('random_forest')

trainer = ModelTrainer()
trained_model = trainer.train(model, X_train, y_train)

# 5. Evaluate the model
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(trained_model, X_train, y_train)
print(f"Model performance: {metrics}")

# 6. Make predictions
predictions = trained_model.predict(X_test)
```

## Project Components

Here's a brief overview of the main components:

### Data Processing

- **DataLoader**: Loads the Titanic dataset from CSV files
- **DataPreprocessor**: Cleans and transforms the data for modeling

### Feature Engineering

- **FeatureCreator**: Creates new features from existing data
- **FeatureSelector**: Selects the most important features

### Modeling

- **ModelFactory**: Creates different types of ML models
- **ModelTrainer**: Trains models and optimizes parameters
- **ModelEvaluator**: Evaluates and compares model performance

### Web Interface

- **App**: Main web application
- **ModelInterface**: Bridge between web and models
- **Dashboard**: Visualizations and interactive components

## Common Tasks

### Training Multiple Models

```python
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer

factory = ModelFactory()
trainer = ModelTrainer()

# Create multiple models
models = {
    'logistic': factory.create_model('logistic_regression'),
    'random_forest': factory.create_model('random_forest'),
    'gradient_boosting': factory.create_model('gradient_boosting')
}

# Train all models
trained_models = {}
for name, model in models.items():
    trained_models[name] = trainer.train(model, X_train, y_train)
    
# Compare models
for name, model in trained_models.items():
    accuracy = model.score(X_val, y_val)
    print(f"{name}: {accuracy:.4f}")
```

### Hyperparameter Tuning

```python
from src.modelling.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = factory.create_model('random_forest')

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Tune hyperparameters
best_model = trainer.tune_hyperparameters(model, X_train, y_train, param_grid)
```

### Generating a Kaggle Submission

```python
from src.utilities.submission_generator import SubmissionGenerator

# Create a submission file
generator = SubmissionGenerator()
submission_df = generator.generate_submission(model, X_test, test_data['PassengerId'])

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Make sure you've installed all requirements with `pip install -r requirements.txt`
2. **Data file not found**: Ensure the `train.csv` and `test.csv` files are in the root directory or `data/` directory
3. **Version conflicts**: The project requires Python 3.8+; check your Python version with `python --version`

### Getting Help

If you encounter any issues:
1. Check the README.md file for detailed documentation
2. Look at the example notebooks in the `notebooks/` directory
3. Submit an issue on GitHub

## Next Steps

Once you're comfortable with the basics, consider:

1. **Improving features**: Experiment with different feature engineering approaches
2. **Trying advanced models**: Test ensemble methods or deep learning models
3. **Optimizing for Kaggle**: Fine-tune your model for better leaderboard performance
4. **Contributing to the project**: Add new features, models, or improvements

Happy predicting!
