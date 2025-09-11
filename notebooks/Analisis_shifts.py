# --- Librerías ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cargar archivo Excel ---
RUTA = "/workspaces/ml-bootcamp-labs/data/raw/Shift_Pozos_LDiscEoc-cvs.csv"   # 👈 cámbialo por la ruta de tu archivo
df = pd.read_excel(RUTA)

# --- Verificación de datos ---
print("Columnas disponibles:", df.columns.tolist())
print(df.head())

# --- Análisis descriptivo general de Delta ---
print("\n📊 Estadísticos generales de Delta:")
print(df['Delta'].describe(percentiles=[0.1, 0.5, 0.9]))

# --- Función de estadísticos personalizados ---
def stats(x):
    return pd.Series({
        'count': x.count(),
        'mean': x.mean(),
        'std': x.std(),
        'min': x.min(),
        'p10': np.percentile(x, 10),
        'p50': np.percentile(x, 50),
        'p90': np.percentile(x, 90),
        'max': x.max()
    })

# --- Agrupación por UWI ---
stats_uwi = df.groupby('UWI')['Delta'].apply(stats).reset_index()
print("\n📌 Estadísticos de Delta por UWI:")
print(stats_uwi)

# --- Agrupación por Common Well Name ---
stats_wellname = df.groupby('Common Well Name')['Delta'].apply(stats).reset_index()
print("\n📌 Estadísticos de Delta por Common Well Name:")
print(stats_wellname)

# --- Visualizaciones ---
plt.figure(figsize=(10,6))
sns.boxplot(x='Common Well Name', y='Delta', data=df)
plt.xticks(rotation=90)
plt.title("Distribución de Delta por Pozo")
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['Delta'], kde=True, bins=30)
plt.title("Histograma de Delta (todos los pozos)")
plt.show()

# --- Análisis espacial ---
plt.figure(figsize=(8,6))
sc = plt.scatter(df['X'], df['Y'], c=df['Delta'], cmap='coolwarm', s=80, edgecolor='k')
plt.colorbar(sc, label="Delta")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Distribución espacial de Delta")
plt.show()
