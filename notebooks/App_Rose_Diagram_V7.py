# App: Diagramas de Rosa con detección de múltiples tendencias
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import itertools
import io
import scipy.signal as signal

# --- Cargar datos ---
RUTA_CSV = '/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv'
df = pd.read_csv(RUTA_CSV, sep=';')

st.title('Generador de Diagramas de Rosa con Tendencias Múltiples')

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

# --- Selectores ---
pozos = sorted(df['wellName'].dropna().unique())
pozo_sel = st.selectbox('Selecciona el pozo:', pozos)

tipos = sorted(df.loc[df['wellName'] == pozo_sel, 'Type'].dropna().unique())
tipo_sel = st.selectbox('Selecciona el tipo de estructura:', tipos)

# Subconjunto
df_tmp = df[(df['wellName'] == pozo_sel) & (df['Type'] == tipo_sel)].copy()

if df_tmp.empty:
    st.error('No hay datos disponibles para la selección actual.')
    st.stop()

# Profundidad disponible
prof_min = float(df_tmp['TDEP-ft'].min())
prof_max = float(df_tmp['TDEP-ft'].max())
st.info(f'Rango disponible para {pozo_sel} - {tipo_sel}: {prof_min:.2f} ft a {prof_max:.2f} ft')

# Entradas de profundidad (número en lugar de slider)
col1, col2 = st.columns(2)
profundidad_min = col1.number_input('Profundidad mínima (ft):', min_value=prof_min, max_value=prof_max, value=prof_min, step=0.1)
profundidad_max = col2.number_input('Profundidad máxima (ft):', min_value=prof_min, max_value=prof_max, value=min(prof_max, prof_min + 100), step=0.1)

if profundidad_min > profundidad_max:
    st.error('La profundidad mínima no puede ser mayor que la máxima.')
    st.stop()

# Parámetros del gráfico
bins = st.slider('Número de bins (divisiones angulares)', 4, 36, 12)
paletas = {
    'Clásico Geológico': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'],
    'Elegante': ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'],
    'Marino/Terrestre': ['#00429d', '#4771b2', '#73a2c6', '#a5d5d8', '#f6f5f5']
}
paleta_sel = st.selectbox('Paleta de colores:', list(paletas.keys()))
colores = list(itertools.islice(itertools.cycle(paletas[paleta_sel]), bins))
modo = st.radio("Modo de diagrama:", ("Orientaciones puras (solo azimut)", "Orientaciones + clasificación por dip"))

# Umbral para detectar picos (fracción del máximo del hist)
umbral_frac = st.slider('Umbral de altura para detectar picos (fracción del máximo)', 0.05, 0.9, 0.4, 0.05)
max_peaks = st.number_input('Número máximo de tendencias a mostrar', min_value=1, max_value=6, value=2, step=1)

# Filtrar por profundidad y limpiar
df_f = df_tmp[(df_tmp['TDEP-ft'] >= profundidad_min) & (df_tmp['TDEP-ft'] <= profundidad_max)].dropna(subset=['Azimuth-dega'])
if df_f.empty:
    st.warning('No hay datos en el intervalo seleccionado.')
    st.stop()

# Cálculo de media vectorial circular (dirección promedio única)
azim_rad = np.deg2rad(df_f['Azimuth-dega'].values)
sin_mean = np.mean(np.sin(azim_rad))
cos_mean = np.mean(np.cos(azim_rad))
R = np.sqrt(sin_mean**2 + cos_mean**2)
mean_angle_deg = (np.degrees(np.arctan2(sin_mean, cos_mean))) % 360

# Preparar figura
fig = plt.figure(figsize=(8,8))
ax = WindroseAxes.from_ax(fig=fig)

# Dibujar barras
if modo == "Orientaciones puras (solo azimut)":
    ax.bar(df_f['Azimuth-dega'], [1]*len(df_f), bins=bins, normed=True, opening=0.8, edgecolor='black', colors=colores)
else:
    ax.bar(df_f['Azimuth-dega'], df_f['Dip_dega'], bins=bins, normed=True, opening=0.8, edgecolor='black', colors=colores)
    ax.set_legend(title="Frecuencia (%)", loc='lower right', bbox_to_anchor=(1.2, 0.1))

# --- Dibujar flecha de media global (invertida +180° para apuntar correctamente) ---
r_max = ax.get_rmax()
ax.arrow(
    np.deg2rad((mean_angle_deg + 180) % 360), 0, 0, r_max * 0.8,
    width=0.05, color='red', alpha=0.9, zorder=10, length_includes_head=True, head_width=0.14
)

# --- Detectar tendencias múltiples (picos en histograma angular) ---
angles = df_f['Azimuth-dega'].values
# usar mismo número de bins que el gráfico para consistencia
hist, bin_edges = np.histogram(angles, bins=bins, range=(0,360))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

if hist.sum() > 0:
    height_threshold = hist.max() * float(umbral_frac)
    peaks, props = signal.find_peaks(hist, height=height_threshold)
    # ordenar picos por altura y limitar a max_peaks
    if peaks.size > 0:
        peak_heights = props['peak_heights']
        order = np.argsort(peak_heights)[::-1]
        selected = order[:int(max_peaks)]
        detected_angles = bin_centers[peaks[selected]]
    else:
        detected_angles = np.array([])
else:
    detected_angles = np.array([])

# Colores para flechas secundarias (evitar rojo, ya usado)
arrow_colors = ['orange', 'purple', 'green', 'magenta', 'cyan']

for i, ang in enumerate(detected_angles):
    ang_plot = (ang + 180) % 360  # invertir para que apunte en sentido geológico correcto
    color = arrow_colors[i % len(arrow_colors)]
    ax.arrow(
        np.deg2rad(ang_plot), 0, 0, r_max * 0.7,
        width=0.04, color=color, alpha=0.9, zorder=9, length_includes_head=True, head_width=0.12
    )

# Añadir leyenda manual para flechas
legend_items = []
legend_labels = []
legend_items.append(plt.Line2D([0],[0], color='red', lw=3))
legend_labels.append(f'Media global {mean_angle_deg:.1f}° (R={R:.2f})')
for i, ang in enumerate(detected_angles):
    legend_items.append(plt.Line2D([0],[0], color=arrow_colors[i % len(arrow_colors)], lw=3))
    legend_labels.append(f'Tendencia {i+1}: {ang:.1f}°')

ax.legend(legend_items, legend_labels, bbox_to_anchor=(1.2, 1.0), loc='upper left')

# Título y subtítulo
plt.title(
    f"Diagrama de Rosa - {pozo_sel} - {tipo_sel}\n"
    f"{profundidad_min:.2f}–{profundidad_max:.2f} ft\n"
    f"Media: {mean_angle_deg:.1f}°  |  R={R:.2f}",
    pad=30
)

fig.subplots_adjust(top=0.85)
st.pyplot(fig)

# Descargar (PNG + JPEG)
buf = io.BytesIO()
fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
buf.seek(0)
st.download_button('Descargar PNG', data=buf.getvalue(), file_name=f'diagrama_rosa_{pozo_sel}_{int(profundidad_min)}-{int(profundidad_max)}ft.png', mime='image/png')

buf_j = io.BytesIO()
fig.savefig(buf_j, format='jpeg', dpi=300, bbox_inches='tight')
buf_j.seek(0)
st.download_button('Descargar JPEG', data=buf_j.getvalue(), file_name=f'diagrama_rosa_{pozo_sel}_{int(profundidad_min)}-{int(profundidad_max)}ft.jpg', mime='image/jpeg')
