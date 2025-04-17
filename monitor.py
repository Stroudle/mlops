import datetime
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from flight_prediction import train
from preprocessing import preprocess_data
from datetime import datetime

# Função para verificar drift
def check_for_drift(drift_score, drift_by_columns):
    num_columns_drift = sum(1 for col, values in drift_by_columns.items() if values.get("drift_detected", False))
    if drift_score > 0.5 or num_columns_drift > 2:
        print(f"Drift detectado! Score de drift: {drift_score}, Colunas com drift: {num_columns_drift}")
        print("Iniciando re-treino do modelo com os dados da API...")
        retrain_model()
    else:
        print("Nenhum drift detectado. O modelo atual ainda é válido.")

# Avaliar degradação do modelo
def evaluate_model(ref_data, current_data):
    # Pré-processar os dados de referência e os dados atuais
    ref_data = preprocess_data(ref_data)
    current_data = preprocess_data(current_data)

    # Avaliar o drift
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=current_data)

    if not os.path.exists("reports"):
        os.makedirs("reports")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/monitoring_report_{timestamp}.html"
    report.save_html(report_path)
    print(f"Relatório salvo em: {report_path}")

    report_dict = report.as_dict()
    drift_score = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Score de drift: {drift_score}")
    drift_by_columns = report_dict["metrics"][1]["result"].get("drift_by_columns", {})
    columns_with_drift = [col for col, values in drift_by_columns.items() if values.get("drift_detected", False)]

    # Imprimir as colunas com drift
    if columns_with_drift:
        print(f"Colunas com drift detectado: {', '.join(columns_with_drift)}")

    return drift_score, drift_by_columns

# Função para carregar dados
def load_data(file_name):
    path = os.path.join("data", f"{file_name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        print("Nenhum dado armazenado encontrado.")
        return None

# Função para re-treinar o modelo com os dados da API
def retrain_model():
    dataset_names = ["Flight_Price_Dataset_of_Bangladesh", "api_inputs"]
    print("Re-treinando o modelo com os dados da API...")
    train(dataset_names)
    print("Re-treino concluído e modelo registrado no MLflow.")

# Função principal
def main():
    df = load_data("Flight_Price_Dataset_of_Bangladesh")
    df_input = load_data("api_inputs")
    drift_score, drift_by_columns = evaluate_model(df, df_input)
    check_for_drift(drift_score, drift_by_columns)

if __name__ == "__main__":
    main()