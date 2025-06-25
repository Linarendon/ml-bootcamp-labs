import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn.sns

# Para una aplicación de streamlit, se corre todo lo que está adentro del objeto "st"
# Títulos
# Por ejemplo, si quiero agregar un título tengo que ponerlo con el método st.title()
st.title("Aplicación de Streamlit")
# Subtitutlos
st.subheader("Visualización de datos")
df = pd.read_csv("data/raw/Operational_events.csv")
st.write("Datos cargados:")
st.dataframe(df)


# Menú de selección
dataframe_numerico = df.select_dtypes(include='number')
columna_seleccionada = st.selectbox("Seleccione una variable: ", df.columns)

# Crear un histograma
fig, ax =plt.subplots(figsize=(12,4))
sns.histplot( df[columna_seleccionada].dropna(), ax=ax )

# Desplegar visualización en la aplicación
st.pyplot(fig)


