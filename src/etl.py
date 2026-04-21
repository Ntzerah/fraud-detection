import pandas as pd
import numpy as np
import os


def carregar_arquivo (caminho):
    df = pd.read_csv(caminho)
    return df

def verificar_dados (df):
    
    print(f"Valores nulos: {df.isnull().sum().sum()}") #Verificação de valores nulos.
    print(f"Valores duplicados: {df.duplicated().sum()}") #Verificação de valores duplicadosdf.head().
    print(f"Transações fraudulentas: {(df['Class'] == 1).sum()}") #Verificar quantas transações são fraudulentas.
    print(df.info())
    print(df.head(5))

def transformar_dados(df):

    df = df.drop_duplicates() #Remoção de duplicatas.
    print(f"Após remover duplicatas: {df.shape[0]} linhas")
    df = df.drop(columns = ['Time']) #Remoção da coluna Time - Não relevante para o modelo.
    #Como o objetivo é a criação de um modelo de Detecção de Fraudes em relação às transações, vamos normalizar a coluna Amount.
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std() #Z-score.

    return df

def salvar_dados(df,caminho_saida):

    os.makedirs(os.path.dirname(caminho_saida), exist_ok = True)# Cria a pasta onde o arquivo será salvo no caminho especificado.
    df.to_csv(caminho_saida, index = False)
    print(f"Dados salvos em: {caminho_saida}")

df = carregar_arquivo('data/raw/creditcard.csv')
verificar_dados(df)
df = transformar_dados(df)
salvar_dados(df,"data/processed/creditcard_limpo.csv")

# verificar_dados(df)
# df = transformar_dados(df)
# salvar_dados(df,"data/processed/creditcard_limpo.csv")





