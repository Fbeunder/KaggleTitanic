# Titanic Survival Prediction

![Titanic Image](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg)

## 📋 Project Overview

The Titanic Survival Prediction project is a machine learning application that analyzes passenger data from the historic Titanic disaster to predict survival outcomes. Based on the famous Kaggle competition dataset, this project demonstrates a complete machine learning workflow including:

1. Data processing and exploratory analysis of the Titanic passenger dataset
2. Feature engineering and selection to improve model performance
3. Training and evaluation of various machine learning models
4. A web interface for data exploration, model comparison, and making predictions
5. Generation of submission files for the Kaggle competition

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended, but optional)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Fbeunder/KaggleTitanic.git
cd KaggleTitanic
```

2. Create and activate a virtual environment (optional but recommended):
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

4. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place the files in the root directory or in the `data/` directory:
   - train.csv
   - test.csv

### Configuration

The application uses a configuration file that can be customized to adjust paths and settings. Configuration is managed through the `src/utilities/config.py` module.

## 🏗️ Project Structure

```
KaggleTitanic/
├── data/                   # Data directory for Titanic datasets
├── notebooks/              # Jupyter notebooks for exploration and analysis
│   ├── titanic_eda.ipynb        # Exploratory Data Analysis notebook
│   └── model_factory_demo.ipynb # Model Factory demonstration
├── src/                    # Source code
│   ├── data_processing/        # Data loading and preprocessing modules
│   │   ├── data_loader.py         # Loading train and test datasets
│   │   └── data_preprocessor.py   # Data cleaning and transformation
│   ├── feature_engineering/     # Feature creation and selection modules
│   │   ├── feature_creator.py     # Feature creation and transformation
│   │   └── feature_selector.py    # Feature selection and importance analysis
│   ├── modelling/               # Model training and evaluation modules
│   │   ├── model_factory.py       # Factory for creating ML models
│   │   ├── model_trainer.py       # Training and tuning models
│   │   └── model_evaluator.py     # Evaluating model performance
│   ├── utilities/               # Utility functions and configuration
│   │   ├── config.py             # Configuration settings
│   │   └── utils.py              # Helper functions
│   └── web_interface/           # Web application interface
│       ├── app.py               # Main Flask application
│       ├── model_interface.py   # Interface between web app and models
│       ├── dashboard.py         # Visualization dashboard
│       ├── templates/           # HTML templates
│       └── static/              # CSS, JavaScript, and images
├── tests/                  # Test modules
├── submissions/            # Generated Kaggle submission files
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🧰 Usage

### Running the Web Interface

To start the web application:

```bash
python -m src.web_interface.app
```

Then open your browser and go to: http://localhost:8050

### Web Interface Features

The web interface provides the following functionality:

1. **Data Exploration:** Visualize and explore the Titanic dataset
2. **Feature Engineering:** Select and create features for model training
3. **Model Training:** Train different models with customizable parameters
4. **Predictions:** Make predictions on the test set or custom input
5. **Kaggle Submissions:** Generate and download submission files

### Training Models Programmatically

You can also train models programmatically:

```python
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.modelling.model_factory import ModelFactory
from src.modelling.model_trainer import ModelTrainer

# Load and preprocess data
loader = DataLoader()
train_data = loader.load_train_data()
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(train_data)

# Create and train a model
factory = ModelFactory()
model = factory.create_model('random_forest')
trainer = ModelTrainer()
trainer.train(model, X, y)

# Make predictions
predictions = model.predict(X_test)
```

### Running Tests

To run tests:

```bash
python -m pytest tests/
```

## 📊 Project Components

### Data Processing

- **DataLoader:** Handles loading the training and testing datasets
- **DataPreprocessor:** Cleans, transforms, and prepares data for modeling

### Feature Engineering

- **FeatureCreator:** Creates new features from existing data
- **FeatureSelector:** Selects the most important features for model training

### Modeling

- **ModelFactory:** Creates different types of machine learning models
- **ModelTrainer:** Trains models with cross-validation and hyperparameter tuning
- **ModelEvaluator:** Evaluates model performance with metrics and visualizations

### Web Interface

- **App:** Main web application using Dash/Flask
- **ModelInterface:** Connects the web interface with the machine learning components
- **Dashboard:** Creates visualizations for the web interface

## 📝 Example Workflows

### Basic Workflow

1. Load and preprocess the data
2. Engineer features
3. Train multiple models
4. Evaluate model performance
5. Generate predictions and submission file

### Advanced Workflow

1. Load and preprocess the data
2. Perform feature engineering with feature importance analysis
3. Use GridSearchCV for hyperparameter tuning
4. Train ensemble models
5. Evaluate with cross-validation
6. Generate predictions with confidence intervals

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## ✉️ Contact

For questions or feedback about this project, please open an issue on GitHub.
