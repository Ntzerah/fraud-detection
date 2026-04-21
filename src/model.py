
#Importando bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

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
def treinar_modelo_random_forest(X_train, y_train):

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def treinar_modelo_xgboost (X_train, y_train):

    modelo_xgb = XGBClassifier(random_state = 42)
    modelo_xgb.fit(X_train,y_train)
    return modelo_xgb

#Avalição do modelo
def avaliar_modelo(modelo, X_test, y_test):
    nome = type(modelo).__name__
    y_pred = modelo.predict(X_test)
    
    print(f"Classification Report: {nome}")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: {nome}\n")
    print(cm)
    
    return y_pred, cm

if __name__ == "__main__":
    X, y = carregar_arquivo(caminho = 'data/processed/creditcard_limpo.csv')
    X_train, X_test, y_train, y_test = dividir_dados(X,y)
    X_res, y_res = balancear_dados(X_train, y_train)
    modelo_randon = treinar_modelo_random_forest(X_res, y_res)
    modelo_xgb = treinar_modelo_xgboost (X_res, y_res)
    avaliar_modelo(modelo_randon, X_test, y_test)
    avaliar_modelo(modelo_xgb, X_test, y_test)
