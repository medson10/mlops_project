# MLOps Final Project

## Dataset used
https://www.kaggle.com/datasets/shree1992/housedata

## Dataset description
The dataset contains information about houses in Sidney and Melbourne. It includes the information about the number of bedrooms,
bathrooms, size, location, price and other features of the houses.

## Requirements

This project can be run using Asdf and Poetry, or using only Python.

## Installation

### Using Asdf and Poetry

1. Install Asdf if you haven't already.
2. Clone the repository.
3. Run `asdf install` this will install the required versions of Python and Poetry.
4. Navigate to the project directory.
5. Run `poetry install` to install the dependencies.

### Using only Python

1. Install Python if you haven't already.
2. Clone the repository.
3. Navigate to the project directory.
4. Run `pip install -r requirements.txt` to install the dependencies.

## Getting Started

- Download the dataset from https://www.kaggle.com/datasets/shree1992/housedata
- Unzip the dataset and place it in the data directory with the name `housing.csv`
- If you are running with Poetry add `poetry run` before the command.
- Start the MLFlow server by running `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Run `python main.py` to train and register the models.
- Serve the model as an API by running `mlflow models serve -m ./mlruns/1/<artifact_id>/artifacts/random_forest_model -p 8000 --no-conda`
- Run `python monitor.py` to load a sample data, check for anomalies and generate a report.
