import itertools
import os
from sklearn.linear_model import LinearRegression
import tempfile
import pandas as pd
import mlflow
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Flight_Price_Prediction")

def preprocess_data(df):
    df = df.drop(columns=['Source Name', 'Destination Name', 'Departure Date & Time', 'Arrival Date & Time'])

    df = df.infer_objects(copy=False)

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('float64')

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])
        df[col] = df[col].astype('float64')

    df.fillna(0, inplace=True)

    X = df.drop(columns=['Total Fare (BDT)'])
    y = df['Total Fare (BDT)']
    
    mlflow_dataset = mlflow.data.from_pandas(df, targets ='Total Fare (BDT)')

    return X, y, mlflow_dataset

def load_data(dataset_name):
    df = pd.read_csv(f"data/{dataset_name}.csv")
    return df

def train(X, y, dataset_name, mlflow_dataset):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'param_grid': {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        'K-NearestNeighbors': {
            'model': KNeighborsRegressor(),
            'param_grid': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }

    for name, params in models.items():
        model = params['model']
        param_grid = params['param_grid']

        keys, values = zip(*param_grid.items())
        for combination in itertools.product(*values):
            param_combo = dict(zip(keys, combination))

            model.set_params(**param_combo)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            run_name = f"{name}_" + "_".join([f"{k}={v}" for k, v in param_combo.items()])

            with mlflow.start_run(run_name=run_name):
                mlflow.log_input(mlflow_dataset, context="training")
                mlflow.log_params(param_combo)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.set_tag("dataset_name", dataset_name)

                signature = infer_signature(X_train, y_pred)
                model_info = mlflow.sklearn.log_model(
                    model,
                    name,
                    signature=signature,
                    input_example=X_train,
                    registered_model_name=f"{name}GridSearch"
                )

                loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
                predictions = loaded_model.predict(X_test)

                result = pd.DataFrame(X_test, columns=X.columns.values)
                result['label'] = y_test.values
                result['predictions'] = predictions

                mlflow.evaluate(
                    data=result,
                    targets="label",
                    predictions="predictions",
                    model_type="regressor"
                )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    result_path = os.path.join(tmp_dir, "predictions.csv")
                    result.to_csv(result_path, index=False)
                    mlflow.log_artifact(result_path, artifact_path="predictions")

def run_model_training(dataset_names):
    data = pd.DataFrame()
    for dataset in dataset_names:
        df = load_data(dataset)
        data = pd.concat([data, df], ignore_index=True)

    X, y, mlflow_dataset = preprocess_data(data)
    train(X, y, dataset_names, mlflow_dataset)

def main():
    dataset_name = ["Flight_Price_Dataset_of_Bangladesh"]
    run_model_training(dataset_name)

if __name__ == "__main__":
    main()