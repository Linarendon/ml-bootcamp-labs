import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes


# --- Cargar datos ---
df = pd.read_csv('/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv', sep=';')

# --- Parámetros de filtrado ---
pozo = 'CARDALES_1N'
prof_min, prof_max = 5590, 5650

# --- Filtrar por pozo y profundidad ---
df_f = df[
    (df['wellName'] == pozo) &
    (df['TDEP-ft'] >= prof_min) &
    (df['TDEP-ft'] <= prof_max)
]

# --- Crear figura y diagrama de rosa ---
fig = plt.figure(figsize=(6, 6))
ax = WindroseAxes.from_ax(fig=fig)

# 'Azimuth_dega' como dirección y 'Dip_dega' como magnitud
ax.bar(
    df_f['Azimuth-dega'],
    df_f['Dip_dega'],
    bins=8,
    normed=True,
    opening=0.8,
    edgecolor='black'
)

# --- Formato ---
ax.set_legend()
plt.title(f'Diagrama de Rosa - {pozo}\n{prof_min}-{prof_max} ft')
plt.show()