# --- App: Diagrama de Rosa + Stereonet ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import mplstereonet
import itertools
import io
import numpy as np

# --- Cargar datos ---
RUTA_CSV = '/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv'
df = pd.read_csv(RUTA_CSV, sep=';')

st.title('Análisis de Orientaciones: Diagrama de Rosa y Stereonet')

# --- Rangos por pozo ---
with st.expander('Ver rangos de profundidad disponibles'):
    rangos_pozo = (
        df.groupby('wellName', as_index=False)['TDEP-ft']
          .agg(prof_min='min', prof_max='max')
          .sort_values('wellName')
    )
    st.dataframe(rangos_pozo)

# --- Selectores ---
pozos = sorted(df['wellName'].dropna().unique())
pozo_sel = st.selectbox('Selecciona el pozo:', pozos)

tipos = sorted(df.loc[df['wellName'] == pozo_sel, 'Type'].dropna().unique())
tipo_sel = st.selectbox('Selecciona el tipo de estructura:', tipos)

df_tmp = df[(df['wellName'] == pozo_sel) & (df['Type'] == tipo_sel)]

if df_tmp.empty:
    st.error('No hay datos disponibles para la selección actual.')
    st.stop()

prof_min = float(df_tmp['TDEP-ft'].min())
prof_max = float(df_tmp['TDEP-ft'].max())

st.info(f'Rango disponible: {prof_min:.2f} – {prof_max:.2f} ft')

# --- Selección del intervalo ---
col1, col2 = st.columns(2)
with col1:
    profundidad_min = st.number_input('Profundidad mínima (ft)', min_value=prof_min, max_value=prof_max, value=prof_min)
with col2:
    profundidad_max = st.number_input('Profundidad máxima (ft)', min_value=prof_min, max_value=prof_max, value=prof_min+100)

if profundidad_min > profundidad_max:
    st.error("La profundidad mínima no puede ser mayor que la máxima.")
    st.stop()

# --- Parámetros de gráfico ---
bins = st.slider('Número de bins (divisiones angulares)', 4, 36, 12)
modo = st.radio('Modo de diagrama:', ['Rosa de Orientaciones', 'Rosa + Dip'])

# --- Paleta ---
paletas = {
    'Clásico': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
    'Elegante': ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'],
}
paleta_sel = st.selectbox('Paleta de colores:', list(paletas.keys()))
colores_final = list(itertools.islice(itertools.cycle(paletas[paleta_sel]), bins))

# --- Filtrado ---
df_f = df_tmp[(df_tmp['TDEP-ft'] >= profundidad_min) & (df_tmp['TDEP-ft'] <= profundidad_max)]

if df_f.empty:
    st.warning("No hay datos en el rango seleccionado.")
    st.stop()

# ==============================================================
# GRÁFICO 1: Diagrama de ROSA
# ==============================================================
fig1 = plt.figure(figsize=(6,6))
ax = WindroseAxes.from_ax(fig=fig1)

if modo == 'Rosa + Dip':
    ax.bar(df_f['Azimuth-dega'], df_f['Dip_dega'], bins=bins, normed=True, opening=0.8,
           edgecolor='black', colors=colores_final)
    plt.title(f"Diagrama de Rosa ({pozo_sel} - {tipo_sel})\n{profundidad_min:.0f}-{profundidad_max:.0f} ft", pad=30)
else:
    ax.bar(df_f['Azimuth-dega'], [1]*len(df_f), bins=bins, normed=True, opening=0.8,
           edgecolor='black', colors=colores_final)
    plt.title(f"Diagrama de Rosa (Orientaciones) - {pozo_sel} - {tipo_sel}\n{profundidad_min:.0f}-{profundidad_max:.0f} ft", pad=30)

st.pyplot(fig1)

# Botón descarga del rosa
buf1 = io.BytesIO()
fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
buf1.seek(0)
st.download_button("Descargar Diagrama de Rosa", buf1, file_name=f"rosa_{pozo_sel}_{tipo_sel}.png", mime="image/png")

# ==============================================================
# GRÁFICO 2: ESTEREONET
# ==============================================================
st.markdown("---")
st.subheader("Proyección Estereográfica (Stereonet)")

# Selector de hemisferio
hemisferio = st.radio(
    "Selecciona el hemisferio de proyección:",
    ("Inferior (estructural, polos a planos)", "Superior (paleocorrientes)"),
    index=0,
)

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.add_subplot(111, projection='stereonet')

# Extraer datos
azimuths = df_f['Azimuth-dega'].to_numpy()
dips = df_f['Dip_dega'].to_numpy()

# Si se selecciona hemisferio superior (paleocorrientes)
if "Superior" in hemisferio:
    # Invertir dirección 180° para reflejar al hemisferio superior
    azimuths_plot = (azimuths + 180) % 360
    label = "Paleocorrientes (Hemisferio superior)"
else:
    # Hemisferio inferior tradicional (estructural)
    azimuths_plot = azimuths
    label = "Polos (Hemisferio inferior)"

# Graficar polos
ax2.pole(azimuths_plot, dips, 'bo', markersize=4, label=label)

# Contornos de densidad si hay suficientes datos
if len(df_f) > 20:
    density = ax2.density_contourf(
        azimuths_plot, dips, measurement='poles', cmap='Blues', alpha=0.6
    )
    fig2.colorbar(density, ax=ax2, orientation='vertical', shrink=0.6, label='Densidad')

# Personalizar título y leyenda
ax2.grid(True)
ax2.legend()
ax2.set_title(
    f"Stereonet - {pozo_sel} - {tipo_sel}\n"
    f"{profundidad_min:.0f}-{profundidad_max:.0f} ft\n{hemisferio}"
)

# Mostrar en Streamlit
st.pyplot(fig2)

# --- Botón de descarga ---
buf2 = io.BytesIO()
fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
buf2.seek(0)
st.download_button(
    "⬇️ Descargar Stereonet",
    buf2,
    file_name=f"stereonet_{pozo_sel}_{tipo_sel}_{'sup' if 'Superior' in hemisferio else 'inf'}.png",
    mime="image/png"
)
