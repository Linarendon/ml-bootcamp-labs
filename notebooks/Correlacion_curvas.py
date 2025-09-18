# --- Librerías ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ruta del archivo ---
RUTA = "/workspaces/ml-bootcamp-labs/data/raw/Comparacion_curvas_Vsh.csv"

# --- Cargar archivo CSV con separador ; ---
df = pd.read_csv(RUTA, sep=";")

# --- Definir intervalo ---
tope = 2000   # reemplaza por tu valor real
base = 2500   # reemplaza por tu valor real

# --- Filtrar por intervalo de profundidad ---
df_intervalo = df[(df["Depth"] >= tope) & (df["Depth"] <= base)]

# --- Pivotear para tener columnas por método ---
df_pivot = df_intervalo.pivot(index="Depth", columns="Metodo", values="Value")

# --- Calcular correlación ---
correlacion = df_pivot.corr().iloc[0,1]
print(f"Correlación entre métodos en el intervalo {tope}-{base}: {correlacion:.4f}")

# --- Graficar curvas ---
plt.figure(figsize=(8,6))
sns.lineplot(data=df_pivot)
plt.title(f"Comparación de curvas entre {tope}-{base} (r={correlacion:.2f})")
plt.xlabel("Profundidad (Depth)")
plt.ylabel("Valor")
plt.legend(title="Método")
plt.gca().invert_yaxis()  # usual en registros de pozo
plt.show()
