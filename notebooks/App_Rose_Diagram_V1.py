# app.py
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import streamlit as st

# --- Cargar datos ---
df = pd.read_csv('/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv', sep=';')

# --- Configuración de la página ---
st.title("Generador de Diagramas de Rosa")

# --- Selección de pozo ---
pozo = st.selectbox("Selecciona el pozo", df['wellName'].unique())

# --- Selección de rango de profundidades ---
prof_min = st.number_input("Profundidad mínima (ft)", min_value=int(df['TDEP_ft'].min()), max_value=int(df['TDEP_ft'].max()), value=5590)
prof_max = st.number_input("Profundidad máxima (ft)", min_value=int(df['TDEP_ft'].min()), max_value=int(df['TDEP_ft'].max()), value=5650)

# --- Botón para generar ---
if st.button("Generar diagrama de rosa"):
    df_f = df[
        (df['wellName'] == pozo) &
        (df['TDEP_ft'] >= prof_min) &
        (df['TDEP_ft'] <= prof_max)
    ]

    if df_f.empty:
        st.warning("No hay datos para ese rango de profundidades.")
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(
            df_f['Azimuth_dega'],
            df_f['Dip_dega'],
            bins=8,
            normed=True,
            opening=0.8,
            edgecolor='black'
        )
        ax.set_legend()
        plt.title(f'Diagrama de Rosa - {pozo}\n{prof_min}-{prof_max} ft')
        st.pyplot(fig)