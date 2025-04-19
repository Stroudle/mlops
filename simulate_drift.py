import numpy as np
import pandas as pd
import requests

# Função para carregar dados
def load_data():
    df = pd.read_csv("data/Flight_Price_Dataset_of_Bangladesh.csv")
    df = df.sample(1000)
    return df

# Função para simular drift nos dados
def simulate_drift(data):
    df = data.copy()
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] += np.random.normal(0, df[col].std() * 0.5, size=len(df))

    return df

# Fazer previsões com o modelo
def get_predictions(data, model):
    columns = [
        "Airline", "Source", "Destination", "Duration (hrs)", "Stopovers",
        "Aircraft Type", "Class", "Booking Source", "Base Fare (BDT)", "Tax & Surcharge (BDT)",
        "Seasonality", "Days Before Departure"
    ]

    # Converter os dados para o formato esperado pelo endpoint
    payload = {
        "model": model,  # Nome do modelo a ser usado
        "data": data[columns].values.tolist(),
        "columns": columns
    }

    # Enviar a requisição para o endpoint do modelo
    url = "http://127.0.0.1:8000/predict"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)

    # Verificar a resposta
    if response.status_code != 200:
        raise ValueError(f"Erro na requisição: {response.status_code}, Detalhes: {response.text}")

    predictions = response.json().get("predictions")
    return predictions

def main():
    data = load_data()
    # Simular drift nos dados
    data = simulate_drift(data)

    model = "linear_regression"
    # Fazer previsões
    predictions = get_predictions(data, model)
    print(f"Predições usando o modelo '{model}':", predictions)

if __name__ == "__main__":
    main()