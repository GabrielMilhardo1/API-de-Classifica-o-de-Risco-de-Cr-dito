'''
Este script realiza o ciclo completo de um projeto de machine learning:
- Carrega e combina dados de duas fontes.
- Realiza engenharia de feature para criar uma variável alvo (risco de crédito).
- Define um pipeline de pré-processamento para dados numéricos e categóricos.
- Treina um modelo de Regressão Logística.
- Avalia o modelo.
- Salva o pipeline completo (pré-processamento + modelo) para uso futuro em uma API.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import sys

def run_training():
    '''Função principal para executar o pipeline de treinamento.'''
    # --- 1. Carga de Dados ---
    print("Carregando os dados...")
    try:
        app_df = pd.read_csv('application_record.csv')
        credit_df = pd.read_csv('credit_record.csv')
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Certifique-se que 'application_record.csv' e 'credit_record.csv' estão no diretório correto.")
        print(e)
        sys.exit(1)

    # --- 2. Engenharia de Feature (Variável Alvo) ---
    # Define um "mau cliente" se ele teve status de dívida de 60 dias ou mais em algum momento.
    # Status: '2': 60-89 dias, '3': 90-119, '4': 120-149, '5': 150+ dias
    print("Realizando engenharia de feature...")
    bad_statuses = ['2', '3', '4', '5']
    credit_df['is_bad_client'] = credit_df['STATUS'].isin(bad_statuses).astype(int)
    
    # Agrupa por cliente para ver se ele já foi um "mau cliente" em algum momento
    client_risk = credit_df.groupby('ID')['is_bad_client'].max().reset_index()

    # --- 3. Combinação e Limpeza dos Dados ---
    print("Combinando e limpando os dados...")
    df = pd.merge(app_df, client_risk, on='ID', how='inner')
    
    # Remove colunas que não serão usadas no modelo
    df = df.drop(['ID', 'FLAG_MOBIL'], axis=1)

    # --- 4. Definição do Pipeline de Pré-processamento ---
    print("Definindo o pipeline de pré-processamento...")
    # Separa features (X) e alvo (y)
    X = df.drop('is_bad_client', axis=1)
    y = df['is_bad_client']

    # Identifica features numéricas e categóricas
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Cria um pipeline para transformar dados numéricos (imputação + normalização)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Cria um pipeline para transformar dados categóricos (imputação + one-hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Usa ColumnTransformer para aplicar as transformações corretas a cada tipo de coluna
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # --- 5. Definição e Treinamento do Modelo ---
    print("Definindo e treinando o modelo...")
    # Define o modelo. class_weight='balanced' ajuda a lidar com o desbalanceamento de classes.
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    # Cria o pipeline final, que primeiro pré-processa os dados e depois treina o modelo
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Divide os dados em conjuntos de treino e teste
    # stratify=y garante que a proporção de bons/maus clientes seja a mesma nos dois conjuntos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Treina o pipeline completo
    pipeline.fit(X_train, y_train)
    print("Treinamento concluído.")

    # --- 6. Avaliação do Modelo ---
    accuracy = pipeline.score(X_test, y_test)
    print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2f}")

    # --- 7. Serialização (Salvando o Pipeline) ---
    print("Salvando o pipeline do modelo...")
    joblib.dump(pipeline, 'modelo_risco_pipeline.joblib')
    print("Pipeline salvo com sucesso em 'modelo_risco_pipeline.joblib'")

if __name__ == '__main__':
    run_training()
