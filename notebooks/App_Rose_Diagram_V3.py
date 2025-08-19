# App_Rose_Diagram_V3.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import itertools
import numpy as np

# --- Cargar datos ---
RUTA_CSV = '/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv'
df = pd.read_csv(RUTA_CSV, sep=';')

st.title('Generador de Diagramas de Rosa')

# --- Rangos por pozo (y por pozo+tipo) ---
with st.expander('Ver rangos de profundidad disponibles'):
    rangos_pozo = (
        df.groupby('wellName', as_index=False)['TDEP-ft']
          .agg(prof_min='min', prof_max='max')
          .sort_values('wellName')
    )
    st.write('Rangos por pozo (ft):')
    st.dataframe(rangos_pozo)

    if 'Type' in df.columns:
        rangos_pozo_tipo = (
            df.groupby(['wellName', 'Type'], as_index=False)['TDEP-ft']
              .agg(prof_min='min', prof_max='max')
              .sort_values(['wellName', 'Type'])
        )
        st.write('Rangos por pozo y tipo (ft):')
        st.dataframe(rangos_pozo_tipo)

# --- Selectores interdependientes ---
pozos = sorted(df['wellName'].dropna().unique())
pozo_sel = st.selectbox('Selecciona el pozo:', pozos)

tipos = sorted(df.loc[df['wellName'] == pozo_sel, 'Type'].dropna().unique())
tipo_sel = st.selectbox('Selecciona el tipo de estructura:', tipos)

# Subconjunto para el pozo+tipo elegidos
df_tmp = df[(df['wellName'] == pozo_sel) & (df['Type'] == tipo_sel)]

if df_tmp.empty or df_tmp['TDEP-ft'].isna().all():
    st.error('No hay datos de profundidad (TDEP-ft) para la selección actual.')
    st.stop()

# Rango real disponible (float para mantener precisión)
prof_min = float(df_tmp['TDEP-ft'].min())
prof_max = float(df_tmp['TDEP-ft'].max())

st.info(f'📌 Rango disponible para {pozo_sel} - {tipo_sel}: '
        f'{prof_min:.2f} ft a {prof_max:.2f} ft')

# --- Entradas numéricas para rango de profundidad ---
col1, col2 = st.columns(2)
with col1:
    profundidad_min = st.number_input(
        'Profundidad mínima (ft):',
        min_value=prof_min,
        max_value=prof_max,
        value=prof_min,
        step=0.1,
        format='%.2f'
    )
with col2:
    # valor por defecto: min + 100 ft o el máximo, lo que ocurra primero
    defecto_max = min(prof_max, prof_min + 100.0)
    profundidad_max = st.number_input(
        'Profundidad máxima (ft):',
        min_value=prof_min,
        max_value=prof_max,
        value=defecto_max,
        step=0.1,
        format='%.2f'
    )

# Validación
if profundidad_min > profundidad_max:
    st.error('La profundidad mínima no puede ser mayor que la máxima.')
    st.stop()

# --- Control de bins ---
bins = st.slider('Número de bins para el diagrama rosa', min_value=4, max_value=36, value=8)

# --- Paletas de colores ---
paletas = {
    'Clásico Geológico': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
    'Elegante': ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'],
    'Marino/Terrestre': ['#00429d', '#4771b2', '#73a2c6', '#a5d5d8', '#f6f5f5']
}
paleta_sel = st.selectbox('Paleta de colores:', list(paletas.keys()))
colores = paletas[paleta_sel]
# Ajustar la paleta al número de bins
colores_final = list(itertools.islice(itertools.cycle(colores), bins))

# --- Filtrado final ---
df_f = df_tmp[
    (df_tmp['TDEP-ft'] >= profundidad_min) &
    (df_tmp['TDEP-ft'] <= profundidad_max)
]

# --- Gráfico ---
if df_f.empty:
    st.warning('⚠️ No hay datos para el rango seleccionado.')
else:
    # Validar columnas de dirección y magnitud
    for col in ['Azimuth-dega', 'Dip_dega']:
        if col not in df_f.columns:
            st.error(f"Falta la columna requerida: '{col}'")
            st.stop()

    fig = plt.figure(figsize=(7, 7))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        df_f['Azimuth-dega'],
        df_f['Dip_dega'],
        bins=bins,
        normed=True,
        opening=0.8,
        edgecolor='black',
        colors=colores_final
    )
    ax.set_legend(title='Frecuencia (%)', loc='lower right', bbox_to_anchor=(1.2, 0.1))
    plt.title(
        f'Diagrama de Rosa - {pozo_sel} - {tipo_sel}\n'
        f'{profundidad_min:.2f}–{profundidad_max:.2f} ft',
        pad=30
    )
    fig.subplots_adjust(top=0.82)
    st.pyplot(fig)
