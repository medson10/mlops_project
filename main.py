import pandas as pd
import mlflow
import mlflow.data
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("house_price_prediction")
dataset_path = "./housing.csv"

def load_data():
    df = pd.read_csv(dataset_path)
    return df

def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(columns=['id', 'date', 'lat', 'long'])
    df = pd.get_dummies(df, columns=['zipcode'])

    X = df.drop(columns=["price"])
    y = df["price"]


    mlflow_dataset = mlflow.data.from_pandas(df, targets="price")
    return X, y, mlflow_dataset

def train_model(X, y, mlflow_dataset):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor()
    lr = LinearRegression()

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    for params in (dict(zip(param_grid.keys(), values)) for values in
                   [(n, d, s) for n in param_grid["n_estimators"]
                               for d in param_grid["max_depth"]
                               for s in param_grid["min_samples_split"]]):

        rf.set_params(**params)
        rf.fit(X_train, y_train)
        y_test_pred = rf.predict(X_test)

        with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}"):
            mlflow.log_input(mlflow_dataset, context="training")
            mlflow.log_params(params)
            mlflow.set_tag("dataset_used", 'housing')

            signature = infer_signature(X_train, y_test_pred)
            model_info = mlflow.sklearn.log_model(rf, "random_forest_model",
                                     signature=signature,
                                     input_example=X_train,
                                     registered_model_name="HousingPriceRandomForest")

            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)
            result = pd.DataFrame(X_test, columns=X.columns)
            result["label"] = y_test
            result["predictions"] = predictions

            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="regressor",
            )

            print(result[:5])

    lr.fit(X_train, y_train)
    lr_y_test_pred = lr.predict(X_test)

    with mlflow.start_run(run_name="LR"):
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.set_tag("dataset_used", 'housing')

        signature = infer_signature(X_train, lr_y_test_pred)
        model_info = mlflow.sklearn.log_model(lr, "linear_regression_model",
                                    signature=signature,
                                    input_example=X_train,
                                    registered_model_name="HousingPriceLinearRegression")

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)
        result = pd.DataFrame(X_test, columns=X.columns)
        result["label"] = y_test
        result["predictions"] = predictions

        mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="regressor",
        )

        print(result[:5])

    return (rf, lr)

def main():
    # Load data
    df = load_data()
    X, y, mlflow_dataset = preprocess_data(df)
    train_model(X, y, mlflow_dataset)


if __name__ == "__main__":
    main()
