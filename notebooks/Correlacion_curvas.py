import pandas as pd
import matplotlib.pyplot as plt

# === 1. Leer el archivo ===
ruta = "/workspaces/ml-bootcamp-labs/data/raw/Comparacion_curvas_Vsh.csv"
df = pd.read_csv(ruta)

# Asegurar que las columnas estén bien leídas
print(df.head())

# === 2. Definir intervalo de interés ===
tope = 2500   # cambia por tu tope
base = 2700   # cambia por tu base

df_intervalo = df[(df["Depth"] >= tope) & (df["Depth"] <= base)]

# === 3. Separar por método ===
df_pivot = df_intervalo.pivot(index="Depth", columns="Metodo", values="Value")

# Revisar que queden dos columnas (una por método)
print(df_pivot.head())

# === 4. Calcular correlación ===
correlacion = df_pivot.corr().iloc[0,1]
print(f"\nCorrelación entre los dos métodos en el intervalo {tope}-{base}: {correlacion:.3f}")

# === 5. Graficar para comparar ===
plt.figure(figsize=(6,8))
plt.plot(df_pivot.index, df_pivot.iloc[:,0], label=df_pivot.columns[0])
plt.plot(df_pivot.index, df_pivot.iloc[:,1], label=df_pivot.columns[1])
plt.gca().invert_yaxis()  # Profundidad hacia abajo
plt.xlabel("Valor")
plt.ylabel("Profundidad (m)")
plt.title(f"Comparación de curvas entre {tope}-{base}\nCorrelación = {correlacion:.3f}")
plt.legend()
plt.grid(True)
plt.show()
