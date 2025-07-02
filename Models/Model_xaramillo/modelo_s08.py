# %% [markdown]
# # Sesión 06 - Manejo de Modelos de Clasificación para entornos operativos

# %% [markdown]
# *Objetivo*
# 
# - Aplicar algoritmos de clasificación para identificar eventos operacionales críticos en la industria petrolera.
# - Evaluar el rendimiento de diferentes clasificadores.
# - Interpretar los resultados y la matriz de confusión.
# - Analizar la eficiencia en pozos productores.

# %%
#!python --version


# %% [markdown]
# ## 0. Carga de Librerías

# %%
# Instalación de dependencias del nuevo codespace
#!pip install -r ../requirements.txt

# %%
# Importación de Librerías
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Modelamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # Modelos de Clasificación
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error # Modelos de Regresión

# %% [markdown]
# ## 1. Carga de Datos

# %%
# Carga desde la carpeta data/raw/

# Ruta Absoluta en Linux o Mac
df = pd.read_csv("/workspaces/ml-bootcamp-labs/data/raw/Operational_events.csv")


# %%
df

# %% [markdown]
# ## 2. EDA (Medidas de Tendencia Central, Análisis de Nulos)

# %%
# Métodos info(), describe()
df.info()

# %%
df.describe()

# %%
df.loc[ df.Date.isnull()  ]

# %%
df.Event_Type.hist()

# %% [markdown]
# ## 3. Transformaciones (Encoding, Imputación)

# %%
df.isnull().sum()

# %%
df.Temperature.describe()

# %%
df.Temperature.to_frame().boxplot()

# %%
# Método de Imputación "Simple"
# Cuando tenemos pocas variables podemos completar el valor faltante con una medida de tendencia central, como la media
df.Temperature.fillna(df.Temperature.mean())
df.Pressure.fillna(df.Pressure.mean())

# %%
from sklearn.preprocessing import LabelEncoder

# %%
label_encoder = LabelEncoder()

# Crear una columna de tipo numérica que va a asignar un número con base a Event_Type
df['Event_Type_n'] = label_encoder.fit_transform(df.Event_Type)

df[["Event_Type_n","Event_Type"]].value_counts()

# %%
# Diccionario Manual para hacer la codificacion
manual_encode = {
    "Normal" : 0,
    "Blockage" : 1,
    "Leak" : 2,
    "Pump Failure": 3
}

df.Event_Type.map(manual_encode)

# %%
df.info()

# %%
# Instrucción para descartar columnas que no sean numéricas
# Y aparte renombra la columna codificada por el nombre original
df_encoded = df.select_dtypes(exclude=['object']).rename(columns={"Event_Type_n":"Event_Type"})

# %%
df_encoded

# %%
# Carga de librería de SimpleImputer
from sklearn.impute import SimpleImputer

# %%
df_encoded.info()

# %%
imputer = SimpleImputer(strategy="mean")

df_imputado = pd.DataFrame( imputer.fit_transform(df_encoded), columns= df_encoded.columns  )

# %%
"""
Event_Type_n  Event_Type  
2             Normal          403
0             Blockage         41
1             Leak             31
3             Pump Failure     25
Name: count, dtype: int64
"""

# %%
df_imputado

# %%
df_imputado.Event_Type.value_counts()

# %%
# Regla de balanceo de clases
(41+31+25)/3 * 1.1

# %%
majority_class_df_sample = df_imputado.loc[ df_imputado.Event_Type == 2 ].sample(35)

# %%
df_imputado.loc[ df_imputado.Event_Type == 2 ].describe()

# %%
majority_class_df_sample.describe()

# %%
majority_class_df_sample

# %%
minority_class_df_sample = df_imputado.loc[ df_imputado.Event_Type != 2]

# %%
minority_class_df_sample

# %%
df_balanced = pd.concat([ majority_class_df_sample, minority_class_df_sample])

# %%
df_balanced.Event_Type.value_counts()

# %% [markdown]
# ## 4. Modelamiento

# %%
# Carga de librerías para modelos
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# %%
X = df_balanced.drop(columns=["Event_Type"])
y = df_balanced.Event_Type
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= 0.8, random_state=23)

# %%
model_dtc = DecisionTreeClassifier()
model_dtc.fit(X_train,y_train)

# %%
model_knc = KNeighborsClassifier()
model_knc.fit(X_train,y_train)

# %%
model_svc = SVC()
model_svc.fit(X_train,y_train)

# %% [markdown]
# ## 5. Evaluación

# %%
predict_dtc = model_dtc.predict(X_test)
predict_knc = model_knc.predict(X_test)
predict_svc = model_svc.predict(X_test)

# %%
# Evaluación DTC
dtc_accuracy = accuracy_score(y_pred= predict_dtc, y_true= y_test) 
dtc_precision = precision_score(y_pred= predict_dtc, y_true= y_test,average='weighted')
dtc_recall = recall_score(y_pred= predict_dtc, y_true= y_test,average='weighted')
dtc_f1 = f1_score(y_pred= predict_dtc, y_true= y_test,average='weighted')
dtc_cmatrix = confusion_matrix(y_pred= predict_dtc, y_true= y_test)

# %%
print(dtc_accuracy)
print(dtc_precision)
print(dtc_recall)
print(dtc_f1)

# %%
dtc_cmatrix

# %%
dtc_cmatrix

# %%
confusion_matrix(y_pred= predict_knc, y_true= y_test)

# %%
confusion_matrix(y_pred= predict_svc, y_true= y_test)


import streamlit as st

st.title("Aplicación de clasificación")

st.write("Modelo de clasificación con DTC")

st.write(dtc_accuracy)