# Titanic Survival Prediction

This project uses machine learning to predict which passengers survived the Titanic shipwreck, based on the famous Kaggle competition dataset.

## Project Overview

The Titanic Survival Prediction project is a machine learning application that:

1. Processes and analyzes the Titanic passenger dataset
2. Creates and selects features to improve model performance
3. Trains and evaluates various machine learning models
4. Provides a web interface for data exploration, model comparison, and making predictions
5. Generates submission files for the Kaggle competition

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
```
git clone https://github.com/Fbeunder/KaggleTitanic.git
cd KaggleTitanic
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place the files in the `data/` directory:
   - train.csv
   - test.csv

## Project Structure

```
KaggleTitanic/
├── data/               # Data directory for Titanic datasets
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Source code
│   ├── data_processing/    # Data loading and preprocessing modules
│   ├── feature_engineering/ # Feature creation and selection modules
│   ├── modelling/           # Model training and evaluation modules
│   ├── utilities/           # Utility functions and configuration
│   └── web_interface/       # Web application interface
├── submissions/        # Generated Kaggle submission files
├── tests/              # Test modules
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Usage

### Running the Web Interface

To start the web application:

```
python -m src.web_interface.app
```

Then open your browser and go to: http://localhost:8050

### Running Tests

To run tests:

```
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
