import pandas as pd
import mlflow
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

def train_random_forest(X, y, dataset_name, mlflow_dataset):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=42)

    for params in (dict(zip(param_grid.keys(), v)) for v in
                   [(n, d, s, l) for n in param_grid['n_estimators']
                                 for d in param_grid['max_depth']
                                 for s in param_grid['min_samples_split']
                                 for l in param_grid['min_samples_leaf']]):
        rf.set_params(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}_{params['min_samples_leaf']}"):	
            mlflow.log_input(mlflow_dataset, context="training")
            mlflow.log_params(params)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.set_tag("dataset_name", dataset_name)

            signature = infer_signature(X_train, y_pred)
            model_info = mlflow.sklearn.log_model(rf, "random_forest_model",
                                     signature=signature,
                                     input_example=X_train, 
                                     registered_model_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}_{params['min_samples_leaf']}")

            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)
            result = pd.DataFrame(X_test, columns=X.columns.values)
            result['label'] = y_test.values
            result['predictions'] = predictions

            print(result.head())

            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="regressor"
            )

    return rf

def main():
    dataset_name = "Flight_Price_Dataset_of_Bangladesh"
    df = load_data(dataset_name)
    X, y, mlflow_dataset = preprocess_data(df)
    rf = train_random_forest(X, y, dataset_name, mlflow_dataset)

if __name__ == "__main__":
    main()