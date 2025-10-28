# Cargar librerías para App de Diagrama de Rosas - Análisis de Paleocorrientes con superposición de tipos

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import itertools
import io

# --- Cargar datos ---
RUTA_CSV = '/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv'
df = pd.read_csv(RUTA_CSV, sep=';')

st.title('Generador de Diagramas de Rosa (Comparativo por Tipo de Estructura)')

# --- Rangos por pozo ---
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

# Mostrar los tipos disponibles en ese pozo
tipos_disponibles = sorted(df.loc[df['wellName'] == pozo_sel, 'Type'].dropna().unique())

# --- Selección múltiple de tipos ---
tipos_sel = st.multiselect(
    'Selecciona uno o dos tipos de estructuras para comparar:',
    tipos_disponibles,
    default=tipos_disponibles[:2]  # por defecto los dos primeros
)

if len(tipos_sel) == 0:
    st.warning("Selecciona al menos un tipo de estructura.")
    st.stop()

# --- Filtrado base ---
df_tmp = df[(df['wellName'] == pozo_sel) & (df['Type'].isin(tipos_sel))]

if df_tmp.empty or df_tmp['TDEP-ft'].isna().all():
    st.error('No hay datos válidos para la selección actual.')
    st.stop()

# Rango real disponible
prof_min = float(df_tmp['TDEP-ft'].min())
prof_max = float(df_tmp['TDEP-ft'].max())

st.info(f'Rango disponible para {pozo_sel}: {prof_min:.2f} ft a {prof_max:.2f} ft')

# --- Entradas de rango ---
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
    defecto_max = min(prof_max, prof_min + 100.0)
    profundidad_max = st.number_input(
        'Profundidad máxima (ft):',
        min_value=prof_min,
        max_value=prof_max,
        value=defecto_max,
        step=0.1,
        format='%.2f'
    )

if profundidad_min > profundidad_max:
    st.error('La profundidad mínima no puede ser mayor que la máxima.')
    st.stop()

# --- Control de bins y paletas ---
bins = st.slider('Número de bins (divisiones angulares)', min_value=4, max_value=36, value=8)

paletas = {
    'Clásico Geológico': ['#003f5c', '#ff6361', '#ffa600', '#bc5090'],
    'Elegante': ['#264653', '#e76f51', '#2a9d8f', '#f4a261'],
    'Marino/Terrestre': ['#00429d', '#ffa600', '#73a2c6', '#bc5090']
}
paleta_sel = st.selectbox('Paleta de colores:', list(paletas.keys()))
colores = paletas[paleta_sel]

# --- Filtrado final ---
df_f = df_tmp[
    (df_tmp['TDEP-ft'] >= profundidad_min) &
    (df_tmp['TDEP-ft'] <= profundidad_max)
]

# --- Modo de visualización ---
modo = st.radio(
    "Modo de diagrama:",
    ("Orientaciones puras (solo azimut)", "Orientaciones + clasificación por dip")
)

# --- Mostrar conteo de medidas ---
if not df_f.empty:
    conteos = df_f['Type'].value_counts().reset_index()
    conteos.columns = ['Tipo de estructura', 'Número de medidas']
    total_medidas = len(df_f)
    st.subheader("📊 Conteo de medidas estructurales")
    st.dataframe(conteos)
    st.success(f"**Total de medidas estructurales en este rango:** {total_medidas}")

# --- Gráfico ---
if df_f.empty:
    st.warning("No hay datos para el rango seleccionado.")
else:
    fig = plt.figure(figsize=(7, 7))
    ax = WindroseAxes.from_ax(fig=fig)

    for i, tipo in enumerate(tipos_sel):
        data_tipo = df_f[df_f['Type'] == tipo]
        if len(data_tipo) == 0:
            continue

        if modo == "Orientaciones puras (solo azimut)":
            ax.bar(
                data_tipo['Azimuth-dega'],
                [1]*len(data_tipo),
                bins=bins,
                normed=True,
                opening=0.8,
                edgecolor='black',
                color=colores[i % len(colores)],
                alpha=0.6,
                label=f"{tipo} (n={len(data_tipo)})"
            )
        else:
            ax.bar(
                data_tipo['Azimuth-dega'],
                data_tipo['Dip_dega'],
                bins=bins,
                normed=True,
                opening=0.8,
                edgecolor='black',
                color=colores[i % len(colores)],
                alpha=0.6,
                label=f"{tipo} (n={len(data_tipo)})"
            )

    plt.title(
        f"Diagrama de Rosa Superpuesto - {pozo_sel}\n"
        f"{profundidad_min:.2f}–{profundidad_max:.2f} ft",
        pad=30
    )
    fig.subplots_adjust(top=0.8)
    st.pyplot(fig)

    # --- Botón de descarga ---
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    st.download_button(
        label="⬇️ Descargar diagrama como PNG",
        data=buffer,
        file_name=f"diagrama_rosa_superpuesto_{pozo_sel}.png",
        mime="image/png"
    )
