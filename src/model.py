
#Importando bibliotecas
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#carregar dados e definir features e target
def carregar_arquivo (caminho):

    df = pd.read_csv(caminho)
    X = df.drop(columns = ['Class'])
    y = df['Class']
    return X, y

#Balancear dados, definindo o conjunto de teste e treino
def dividir_dados (X,y):

    X_train, X_test, y_train, y_test = train_test_split (
        X,
        y,
        test_size= 0.2,
        random_state= 42
    )
    return X_train, X_test, y_train, y_test

#Balancear os dados com SMOTE (Gerar registros com padrões semelhantes às fraudes para equilibrar as Classes)
def balancear_dados(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(X_train,y_train)
    return X_res, y_res

#Treinar o modelo
def treinar_modelo(X_train, y_train):

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

#Avalição do modelo
def avaliar_modelo(modelo, X_test, y_test):

    #Previsão
    y_pred = modelo.predict(X_test)

    #Relatório de classificação
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    #Resumo do desempenho do modelo
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    X, y = carregar_arquivo(caminho = 'data/processed/creditcard_limpo.csv')
    X_train, X_test, y_train, y_test = dividir_dados(X,y)
    X_res, y_res = balancear_dados(X_train, y_train)
    modelo = treinar_modelo(X_res, y_res)
    avaliar_modelo(modelo, X_test, y_test)
