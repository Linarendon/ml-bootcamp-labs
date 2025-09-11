import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import numpy as np

# El objetivo de esta aplicación es:
# 1. cargar un modelo generado previamente <- pickle
# 2. recuperar los datos operativos del usuario <- streamlit 
# 3. validar los datos (identificar se sean valores numéricos) <- built-in
# 4. hacer la predicción de estos valores <- model.predict
# 5. devolver al usuario, el estado operativo (Blockage, Leakage, Normal, Pump Failure <- remapeo


# --- Implementación --- #
# 1. Carga del modelo
with open('/workspaces/ml-bootcamp-labs/models/model_op.pkl', 'rb') as file:
    model = pickle.load(file)


# Función de predicción

def prediccion_datos(campos,valores):
    diccionario = dict(zip(campos,valores))
    data = pd.DataFrame(diccionario,index=[0])
    # 3. Validación de datos
    # Well_ID {Los valores válidos para well ID son los pozos: 1 al 20}
    pozos_validos = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # Hay una forma más elegante que es con range(1,21,1)
    if data['Well_ID'][0] not in pozos_validos:
        st.error(f"Pozo {data.Well_ID.values} no es un valor válido (1-20)")
        return None
    # Mantenimiento
    mantenimiento_validos = [0,1]
    if data['Maintenance_Required'][0] not in mantenimiento_validos:
        st.error(f"Valor en Maintenance_Required no es un valor válido (0,1)")
        return None
    else:
        return model.predict(data)

def remapeo_resultados(resultado):
    res_diccionario = {
        0:'Blockage',
        1:'Leakage',
        2:'Normal',
        3:'Pump Failure'
    }
    return res_diccionario[int(resultado)]

# 2. Carga de datos

# En la lista de campos de entrada, se ponen los nombres de las columnas en el orden que las necesitamos
campos = ['Well_ID', 'Pressure', 'Temperature', 'Flow_Rate', 'Pump_Speed',
       'Gas_Oil_Ratio', 'Water_Cut', 'Vibration', 'Maintenance_Required',
       'Downtime'] 
# En la lista de valores, esta se deja vacía porque la aplicación se encarga de llenarlas
valores = []

st.markdown("Introduce los datos de operación para predecir:")

# Ciclo: Se encarga de desplegar un formulario en la aplicación
for c in campos:
    v = st.number_input(f"{c} (numérico)")
    valores.append(v)


# 4. Predicción
if st.button("Predecir!"):
    # 3. Validación
    resultado = prediccion_datos(campos,valores)
    if resultado is None:
        st.error("No se pudo completar la predicción")
    else:    
        st.success(f"Status operativo: {remapeo_resultados(resultado)}")

# 5. Menú de selección

df = pd.read_csv("data/raw/Operational_events.csv")

st.write("Datos cargados:")
st.dataframe(df)

dataframe_numerico = df.select_dtypes(include='number')
columna_seleccionada = st.selectbox("Selecciona una variable: ", dataframe_numerico.columns)
# Creación de un histograma
fig, ax = plt.subplots(figsize=(12,4))
sns.histplot( df[columna_seleccionada].dropna(), ax=ax )
# Desplegar visualización en la aplicación
st.pyplot(fig)

# 6.Fondo personalizado

st.markdown("""
            <style>
            body {
                background-color: #7c1dbf;
            }
            .stApp {
                background-image: url('https://picsum.photos/120/800');
                background-size: cover;
            }
            </style>
            """, unsafe_allow_html=True)


# 7. Diseno y layout

st.sidebar.title("Menú lateral")
st.sidebar.selectbox("Opción", ["Inicio", "Datos", "Predicciones"])

col1, col2 = st.columns (2)
with col1:
        st.write("Columna 1")
with col2:
        st.write("Columna 2")

with st.expander("Ver más detalles") :
     st.write ("Aquí puedes esconder contenido.")


# 8. Cargar una imagen

from PIL import Image
img = Image.open("/workspaces/ml-bootcamp-labs/data/raw/EOR.png")
st.image(img, caption="Logo", use_container_width=True)


