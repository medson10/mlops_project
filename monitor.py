import pandas as pd
import numpy as np
import requests
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
import os
import json

dataset_path = './housing.csv'

def check_for_drift(drift_share):
    if drift_share > 0.5:
        print("Drift detectado no Dataset - retreinando modelo")
        os.system("python3 main.py")
    else:
        print("Nenhum drift detectado")

def load_new_data():
    df = pd.read_csv(dataset_path)
    df = df.sample(1000)  # Pegamos exemplos aleatórios para testar
    X, y = preprocess_data(df)
    return X, y

def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(columns=['id', 'date', 'lat', 'long'])
    df = pd.get_dummies(df, columns=['zipcode'])
    X = df.drop(columns=["price"])
    y = df["price"]

    print(df.head())

    return X, y.astype(int)

# Fazer previsões com o modelo
def get_predictions(data):
    print(data.head())

    # Defina as colunas esperadas pelo modelo
    columns = [
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'waterfront',
        'view',
        'condition',
        'grade',
        'sqft_above',
        'sqft_basement',
        'yr_built',
        'yr_renovated',
        'sqft_living15',
        'sqft_lot15',
        'zipcode_98001', 'zipcode_98002', 'zipcode_98003', 'zipcode_98004', 'zipcode_98005', 'zipcode_98006', 'zipcode_98007', 'zipcode_98008', 'zipcode_98010', 'zipcode_98011', 'zipcode_98014', 'zipcode_98019', 'zipcode_98022', 'zipcode_98023', 'zipcode_98024', 'zipcode_98027', 'zipcode_98028', 'zipcode_98029', 'zipcode_98030', 'zipcode_98031', 'zipcode_98032', 'zipcode_98033', 'zipcode_98034', 'zipcode_98038', 'zipcode_98039', 'zipcode_98040', 'zipcode_98042', 'zipcode_98045', 'zipcode_98052', 'zipcode_98053', 'zipcode_98055', 'zipcode_98056', 'zipcode_98058', 'zipcode_98059', 'zipcode_98065', 'zipcode_98070', 'zipcode_98072', 'zipcode_98074', 'zipcode_98075', 'zipcode_98077', 'zipcode_98092', 'zipcode_98102', 'zipcode_98103', 'zipcode_98105', 'zipcode_98106', 'zipcode_98107', 'zipcode_98108', 'zipcode_98109', 'zipcode_98112', 'zipcode_98115', 'zipcode_98116', 'zipcode_98117', 'zipcode_98118', 'zipcode_98119', 'zipcode_98122', 'zipcode_98125', 'zipcode_98126', 'zipcode_98133', 'zipcode_98136', 'zipcode_98144', 'zipcode_98146', 'zipcode_98148', 'zipcode_98155', 'zipcode_98166', 'zipcode_98168', 'zipcode_98177', 'zipcode_98178', 'zipcode_98188', 'zipcode_98198', 'zipcode_98199', 'zipcode_98039', 'zipcode_98039'
    ]

    # Crie uma lista de dicionários, onde cada dicionário representa uma instância
    instances = []
    for _, row in data.iterrows():
        instance = {col: row[col] for col in columns}
        instances.append(instance)


    url = "http://127.0.0.1:8000/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": instances}

    response = requests.post(url, headers=headers, json=payload)
    predictions = response.json()
    predictions = predictions.get("predictions")
    return predictions

# Avaliar degradação do modelo
def evaluate_model(df, y):
    df["prediction"] = get_predictions(df)
    df["target"] = y
    data_definition = DataDefinition(
        regression=[Regression(target=df["target"].name, prediction=df["prediction"].name)]
    )
    dataset = Dataset.from_pandas(
        df,
        data_definition=data_definition,
    )

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    my_eval = report.run(reference_data=dataset, current_data=dataset)
    my_eval.save_html("monitoring_report_df.html")
    report_dict = my_eval.dict()
    drift_share = report_dict["metrics"][0]["value"]["share"]
    return drift_share

def main():
    df_examples, y = load_new_data()
    drift_share = evaluate_model(df_examples, y)
    check_for_drift(drift_share)

if __name__ == "__main__":
    main()
