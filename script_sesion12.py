# Implementación de un código para generar modelos a partir de datos operativos.

# 0. Cargar librerías
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score

# 1. Cargar datos

df = pd.read_csv('data/raw/Operational_events.csv')

# 2. EDA de datos

# En esta ocasión no haremos EDA

# 3. Transformaciones
# 3.1 Reducción del DF

df = df.drop(columns=['Date','Time'])

# 3.2 LabelEncoding

le = LabelEncoder()
df['Event_Type'] = le.fit_transform(df['Event_Type'])

# 3.3 Normalización (MinMax)

X_prescale = df.drop(columns=['Event_Type'])

scaler = MinMaxScaler()

X = scaler.fit_transform(X_prescale)
X = pd.DataFrame(X, columns=X_prescale.columns)

y = df['Event_Type']


# 4. --- Modelado ---

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=23, stratify=y)

# Implementación de Random Forest
rf_model = RandomForestClassifier(
    n_estimators=1000 , # Número de árboles de decisión
    max_depth=10 , # Límite de modelo
    random_state=23, # Semilla para replicas
    n_jobs=-1 # Nucleos de procesador
)

rf_model.fit(X_train,y_train)

rf_y_pred = rf_model.predict(X_test)

print('Model RF Done!')

# Implementación de XGBoost

xgb_model = XGBClassifier(
    n_estimators= 1000,
    max_depth=10,
    random_state=23,
    learning_rate=0.1,
    eval_metric='mlogloss',
)

xgb_model.fit(X_train,y_train)

xgb_y_pred = xgb_model.predict(X_test)

print('Model XGB Done!')

# 5. Evaluación

def model_evaluation(nombre,y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred,average='weighted')
    f1 = f1_score(y_true,y_pred,average='weighted')

    print(f"Resultados para el modelo: {nombre}\n")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tRecall: {recall}")
    print(f"\tF1: {f1}")

model_evaluation("Random Forest",y_test,rf_y_pred)
model_evaluation("XGBoost", y_test,xgb_y_pred)

print('OK')