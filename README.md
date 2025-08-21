# API de Classificação de Risco de Crédito

## 🎯 Objetivo

Esta API fornece uma solução completa para classificação de risco de crédito baseada em Machine Learning. O projeto transforma um modelo treinado em um serviço web acessível, capaz de prever se um cliente representa um "ALTO RISCO" ou "BAIXO RISCO" com base em seus dados.

A API foi construída com FastAPI e containerizada com Docker, garantindo portabilidade e facilidade de implantação.

## ✨ Features

- **Predição de Risco**: Endpoint `/predict` para classificações em tempo real.
- **Probabilidades**: Fornece não apenas a classificação, mas a probabilidade de cada classe.
- **Containerização**: Empacotado com Docker para uma execução consistente em qualquer ambiente.
- **Pipeline Completo**: Inclui scripts para análise, treinamento e serialização do modelo.

## 🛠️ Tecnologias Utilizadas

- **Python 3.9**
- **FastAPI**: Para a construção da API web.
- **Scikit-learn**: Para o treinamento e pipeline do modelo de Machine Learning.
- **Pandas**: Para manipulação de dados.
- **Joblib**: Para serialização do pipeline do modelo.
- **Docker**: Para a containerização da aplicação.
- **Uvicorn**: Como servidor ASGI para a API.

## 🚀 Como Executar o Projeto

Para executar esta API, você precisa ter o **Docker Desktop** instalado e em execução.

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/seu-usuario/API-de-Classifica-o-de-Risco-de-Cr-dito.git
    cd API-de-Classifica-o-de-Risco-de-Cr-dito
    ```

2.  **Construa a Imagem Docker:**
    O `Dockerfile` presente na raiz do projeto contém todas as instruções para construir a imagem da aplicação.
    ```bash
    docker build -t credit-risk-api .
    ```

3.  **Execute o Contêiner Docker:**
    Este comando iniciará o contêiner e mapeará a porta 8000, tornando a API acessível.
    ```bash
    docker run -d -p 8000:8000 credit-risk-api
    ```

4.  **Verifique se o Contêiner está Rodando:**
    ```bash
    docker ps
    ```
    Você deverá ver o contêiner `credit-risk-api` na lista.

##  kullanım API

A API possui um endpoint principal para predições: `POST /predict`.

Você pode testá-lo usando `curl` ou qualquer cliente de API.

**Exemplo de Requisição com `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "CODE_GENDER": "M",
  "FLAG_OWN_CAR": "Y",
  "FLAG_OWN_REALTY": "Y",
  "CNT_CHILDREN": 0,
  "AMT_INCOME_TOTAL": 427500.0,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Civil marriage",
  "NAME_HOUSING_TYPE": "Rented apartment",
  "DAYS_BIRTH": -12005,
  "DAYS_EMPLOYED": -4542,
  "FLAG_WORK_PHONE": 1,
  "FLAG_PHONE": 0,
  "FLAG_EMAIL": 0,
  "OCCUPATION_TYPE": "Managers",
  "CNT_FAM_MEMBERS": 2.0
}'
```

**Exemplo de Resposta:**

```json
{
  "prediction": "BAIXO RISCO",
  "probability_baixo_risco": "0.70",
  "probability_alto_risco": "0.30"
}
```

## 📁 Estrutura do Projeto

```
.
├── .gitignore
├── analise_e_treinamento.py  # Script para análise e treinamento do modelo
├── application_record.csv    # Dados de aplicação dos clientes
├── credit_record.csv         # Histórico de crédito dos clientes
├── Dockerfile                # Define a imagem Docker da aplicação
├── main.py                   # Lógica da API com FastAPI
├── modelo_risco_pipeline.joblib # Pipeline do modelo serializado
├── README.md                 # Documentação do projeto
└── requirements.txt          # Dependências Python
```