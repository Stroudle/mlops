from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

MODEL_NAME = "RandomForestGridSearch"
MODEL_STAGE = "Production"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Definir o esquema de entrada
class PredictionRequest(BaseModel):
    data: list
    columns: list

# Endpoint para verificar o status da API
@app.get("/")
def read_root():
    return {"status": "API online."}

# Endpoint para previsões
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame(request.data, columns=request.columns)
        
        predictions = model.predict(input_data)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))