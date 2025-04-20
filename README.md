# MLOps Pipeline

Este repositório contém um pipeline de MLOps para monitoramento, re-treino e implantação de modelos de aprendizado de máquina.

---

## **1. Estrutura do Projeto**

```
mlops/
├── app.py               # API para previsões
├── train_model.py       # Treinamento e re-treino do modelo
├── monitor.py           # Monitoramento de drift e re-treino
├── simulate_drift.py    # Simulação de drift nos dados
├── promote.py           # Promoção de modelos no MLflow
├── preprocess.py        # Funções de pré-processamento
├── requirements.txt     # Dependências do projeto
└── data/                # Dados de entrada
```

---

## **2. Pré-requisitos**

- **Python 3.10**
- **MLflow** (para rastreamento de experimentos)

---

## **3. Configuração**

### **Instalação Local**

1. Clone o repositório:
   ```bash
   git clone https://github.com/Stroudle/mlops.git
   cd mlops
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Inicie a base de dados:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

4. Treine os modelos:
   ```bash
   python train_model.py
   python promote.py
   ```

5. Faça previsões via API:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000
   python simulate_drift.py
   ```

6. Monitore o desempenho do modelo:
   ```bash
   python monitor.py
   ```

---

## **4. Funcionalidades**

### **Treinamento**
- O pipeline suporta o treinamento de dois modelos de regressão:
    1. Linear Regression:
        - Modelo simples e interpretável para prever valores contínuos.
        - Suporte a otimização de hiperparâmetros `fit_intercept` , `positive`.
    2. K-Nearest Neighbors (KNN):
        - Modelo baseado em proximidade para prever valores contínuos.
        - Suporte a otimização de hiperparâmetros como número de vizinhos (`n_neighbors`), pesos (`weights`) e métrica de distância (`metric`).

### **Promoção de Modelos**
- Avalia todas as versões dos modelos registrados no MLflow.
- Promove modelos para o estágio Staging com base em um limite de R-squared.
- Identifica o melhor modelo geral entre todos os modelos e promove para o estágio `Production`.

### **API**
- Endpoint `/predict` para previsões em tempo real.

### **Monitoramento**
- Detecta *drift* nos dados usando Evidently.
- Gera relatórios em HTML na pasta `reports/`.

### **Re-treino**
- Combina os dados originais e os dados recebidos via API para re-treinar o modelo.
- Registra o novo modelo no MLflow.

---

## **5. Estrutura de Dados**

Os dados de entrada devem conter as seguintes colunas:

- `Airline`
- `Source`
- `Destination`
- `Duration (hrs)`
- `Stopovers`
- `Aircraft Type`
- `Class`
- `Booking Source`
- `Base Fare (BDT)`
- `Tax & Surcharge (BDT)`
- `Seasonality`
- `Days Before Departure`