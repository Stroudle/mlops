from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from preprocessing import preprocess_data

app = FastAPI()

MODEL_NAME = "RandomForestGridSearch"
MODEL_STAGE = "Production"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Definir o esquema de entrada
class PredictionRequest(BaseModel):
    data: list
    columns: list

# Função para salvar os dados de entrada
def save_input_data(input_data: pd.DataFrame):
    file_path = "data/api_inputs.csv"
    if not os.path.exists("data"):
        os.makedirs("data")
    if os.path.exists(file_path):
        input_data.to_csv(file_path, mode="a", header=False, index=False)
    else:
        input_data.to_csv(file_path, mode="w", header=True, index=False)

# Endpoint para verificar o status da API
@app.get("/")
def read_root():
    return {"status": "API online."}

# Endpoint para previsões
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame(request.data, columns=request.columns)
        
        save_input_data(input_data)

        input_data = preprocess_data(input_data)

        predictions = model.predict(input_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))