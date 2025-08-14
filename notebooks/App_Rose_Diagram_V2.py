# App_Rose_Diagram_V1.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes

# --- Cargar datos ---
RUTA_CSV = '/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv'

df = pd.read_csv(RUTA_CSV, sep=';')

# --- Interfaz de usuario ---
st.title("Generador de Diagramas de Rosa")

# Selector de pozo
pozos = sorted(df['wellName'].unique())
pozo_sel = st.selectbox("Selecciona el pozo:", pozos)

# Rango de profundidades
prof_min = int(df['TDEP-ft'].min())
prof_max = int(df['TDEP-ft'].max())
rango_prof = st.slider(
    "Rango de profundidad (ft):",
    min_value=prof_min,
    max_value=prof_max,
    value=(prof_min, prof_min + 100)
)

# --- Filtrar datos ---
df_f = df[
    (df['wellName'] == pozo_sel) &
    (df['TDEP-ft'] >= rango_prof[0]) &
    (df['TDEP-ft'] <= rango_prof[1])
]

# --- Graficar diagrama de rosa ---
if not df_f.empty:
    fig = plt.figure(figsize=(6, 6))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        df_f['Azimuth-dega'],
        df_f['Dip_dega'],
        bins=8,
        normed=True,
        opening=0.8,
        edgecolor='black'
    )
    ax.set_legend()
    plt.title(f'Diagrama de Rosa - {pozo_sel}\n{rango_prof[0]}-{rango_prof[1]} ft')

    st.pyplot(fig)
else:
    st.warning("No hay datos para el pozo y rango de profundidad seleccionados.")