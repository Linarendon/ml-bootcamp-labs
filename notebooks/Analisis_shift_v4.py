# ============================
# 1. Importar librerías
# ============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 2. Cargar la tabla
# ============================
# Reemplaza con el nombre de tu archivo
df = pd.read_excel (/workspaces/ml-bootcamp-labs/data/raw/Shift_Pozos_LDiscEoc-cvs.csv)

# Ver columnas y primeras filas
print(df.head())

# ============================
# 3. Estadísticos descriptivos
# ============================
# Para Delta y Absolute Delta
stats = df[["Delta", "Absolute Delta"]].describe(percentiles=[0.25,0.5,0.75])
print(stats)

# ============================
# 4. Histogramas de distribución
# ============================
plt.figure(figsize=(8,5))
sns.histplot(df["Delta"], bins=20, kde=True)
plt.title("Distribución de Delta (Shift)")
plt.xlabel("Delta (ft o ms)")
plt.ylabel("Frecuencia")
plt.show()

# ============================
# 5. Boxplot para detectar outliers
# ============================
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Delta"])
plt.title("Boxplot de Delta (Shift)")
plt.xlabel("Delta")
plt.show()

# ============================
# 6. Análisis espacial (si tienes X, Y)
# ============================
plt.figure(figsize=(8,6))
sc = plt.scatter(df["X"], df["Y"], c=df["Delta"], cmap="RdBu", s=60, edgecolor="k")
plt.colorbar(sc, label="Delta")
plt.title("Distribución espacial de Delta")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ============================
# 7. Correlación con profundidad
# ============================
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Pick Z"], y=df["Delta"])
plt.title("Delta vs Profundidad (Pick Z)")
plt.xlabel("Profundidad (Pick Z)")
plt.ylabel("Delta")
plt.axhline(0, color="red", linestyle="--")
plt.show()

# ============================
# 8. Exportar resultados
# ============================
# Guardar estadísticas a Excel
stats.to_excel("estadisticas_shifts.xlsx")
