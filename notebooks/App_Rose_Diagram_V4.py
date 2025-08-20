import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplstereonet

# Cargar datos
df = pd.read_csv('/workspaces/ml-bootcamp-labs/data/raw/DATOS_DIAGRAMA_ROSA_UNIFICADO.csv', sep=';')

# Filtrar por pozo y profundidad
pozo = 'CARDALES_1N'
prof_min, prof_max = 5590, 5650
df_f = df[(df['wellName'] == pozo) &
          (df['TDEP-ft'] >= prof_min) &
          (df['TDEP-ft'] <= prof_max)]

# Convertir rumbo a radianes
trend = np.deg2rad(df_f['Azimuth-dega'])

# Crear histograma polar tipo rosa
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='polar')

# Histograma
n, bins, patches = ax.hist(trend, bins=16, edgecolor='black')

# Formato
ax.set_theta_zero_location('N')  # Norte arriba
ax.set_theta_direction(-1)       # Sentido horario
ax.set_title(f'Diagrama de Rosa - {pozo}\n{prof_min}-{prof_max} ft')

plt.show()