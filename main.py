'''
API para Classificação de Risco de Crédito

Este script usa o FastAPI para criar um servidor web que expõe um modelo de 
machine learning treinado para prever o risco de crédito de um cliente.

Endpoints:
- GET /: Retorna uma mensagem de boas-vindas.
- POST /predict: Recebe os dados de um cliente em formato JSON e retorna a 
  previsão de risco ('ALTO RISCO' ou 'BAIXO RISCO').
'''

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Inicialização da API ---
app = FastAPI(
    title="API de Classificação de Risco de Crédito",
    description="API para prever o risco de crédito de clientes com base em seus dados.",
    version="1.0.0"
)

# --- 2. Carregamento do Modelo ---
# O pipeline do modelo é carregado uma única vez no início para maior eficiência.
try:
    pipeline = joblib.load('modelo_risco_pipeline.joblib')
    print("Pipeline do modelo carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_risco_pipeline.joblib' não encontrado.")
    pipeline = None

# --- 3. Definição do Modelo de Dados (Pydantic) ---
# Define a estrutura e os tipos de dados esperados na requisição POST /predict
# Estes campos devem corresponder às colunas usadas no treinamento do modelo.
class ClientData(BaseModel):
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str
    CNT_FAM_MEMBERS: float

# --- 4. Definição dos Endpoints ---

@app.get("/")
def read_root():
    '''Endpoint raiz que retorna uma mensagem de boas-vindas.'''
    return {"message": "Bem-vindo à API de Classificação de Risco de Crédito"}

@app.post("/predict")
def predict(data: ClientData):
    '''
    Endpoint de predição. Recebe dados do cliente e retorna a classificação de risco.
    '''
    if not pipeline:
        return {"error": "Modelo não carregado. Verifique os logs do servidor."}

    # Converte os dados recebidos (Pydantic model) para um DataFrame do Pandas
    # O pipeline espera um DataFrame como entrada.
    input_df = pd.DataFrame([data.dict()])

    # Faz a predição
    # O pipeline aplica todas as etapas de pré-processamento e depois a classificação.
    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)

    # Mapeia o resultado numérico para uma string descritiva
    risk_mapping = {0: 'BAIXO RISCO', 1: 'ALTO RISCO'}
    prediction_text = risk_mapping.get(prediction[0], "desconhecido")
    
    # Retorna o resultado
    return {
        "prediction": prediction_text,
        "probability_baixo_risco": f"{probability[0][0]:.2f}",
        "probability_alto_risco": f"{probability[0][1]:.2f}"
    }
