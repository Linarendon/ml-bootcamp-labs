# App_Rose_Diagram_V3.py

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

# Filtrar tipos disponibles según el pozo
tipos = sorted(df[df['wellName'] == pozo_sel]['Type'].dropna().unique())
tipo_sel = st.selectbox("Selecciona el tipo de estructura:", tipos)

# Filtrar profundidades disponibles según pozo y tipo
df_filtrado_tmp = df[(df['wellName'] == pozo_sel) & (df['Type'] == tipo_sel)]
prof_min = int(df_filtrado_tmp['TDEP-ft'].min())
prof_max = int(df_filtrado_tmp['TDEP-ft'].max())

# Mostrar info de profundidades disponibles
st.info(f"📌 Profundidad disponible para {pozo_sel} - {tipo_sel}: {prof_min} ft a {prof_max} ft")

# --- Entradas numéricas para rango de profundidad ---
profundidad_min = st.number_input(
    "Profundidad mínima (ft):",
    min_value=prof_min,
    max_value=prof_max,
    value=prof_min
)

profundidad_max = st.number_input(
    "Profundidad máxima (ft):",
    min_value=prof_min,
    max_value=prof_max,
    value=min(prof_max, prof_min + 100)
)

# --- Validación ---
if profundidad_min > profundidad_max:
    st.error("⚠️ La profundidad mínima no puede ser mayor que la máxima.")
    st.stop()

# --- Filtrar datos definitivos ---
df_f = df_filtrado_tmp[
    (df_filtrado_tmp['TDEP-ft'] >= profundidad_min) &
    (df_filtrado_tmp['TDEP-ft'] <= profundidad_max)
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
    plt.title(
        f"Diagrama de Rosa - {pozo_sel} - {tipo_sel}\n{profundidad_min}-{profundidad_max} ft"
    )

    fig.subplots_adjust(top=0.8)  # deja espacio para el título

    st.pyplot(fig)
else:
    st.warning("⚠️ No hay datos para la selección realizada.")
