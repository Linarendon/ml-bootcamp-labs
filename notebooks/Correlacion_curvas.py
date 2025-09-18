import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Cargar datos
file_path = "/workspaces/ml-bootcamp-labs/data/raw/Comparacion_curvas_Vsh.csv"
df = pd.read_csv(file_path, sep=";")

# Pivotear datos para tener columnas por método
df_pivot = df.pivot(index="Depth", columns="Metodo", values="Value").reset_index()
metodos = df["Metodo"].unique()

# Sidebar para definir intervalo
st.sidebar.header("Definir intervalo de profundidad")
prof_min, prof_max = float(df["Depth"].min()), float(df["Depth"].max())

# Slider
tope, base = st.sidebar.slider(
    "Selecciona el rango de profundidad",
    min_value=prof_min,
    max_value=prof_max,
    value=(prof_min, prof_max)
)

# Entradas manuales
tope_input = st.sidebar.number_input("Tope (ft)", value=tope, min_value=prof_min, max_value=prof_max)
base_input = st.sidebar.number_input("Base (ft)", value=base, min_value=prof_min, max_value=prof_max)

# Validación de intervalo
if tope_input >= base_input:
    st.sidebar.error("⚠️ El tope debe ser menor que la base")
else:
    tope, base = tope_input, base_input

# Filtrar datos
df_f = df_pivot[(df_pivot["Depth"] >= tope) & (df_pivot["Depth"] <= base)].dropna()

st.title("Comparación de curvas por métodos")
st.write(f"Intervalo seleccionado: **{tope:.2f} - {base:.2f} ft**")

# Calcular correlación
corr, _ = pearsonr(df_f[metodos[0]], df_f[metodos[1]])
st.write(f"📊 Correlación de Pearson entre los métodos: **{corr:.3f}**")

# Gráfico de dispersión con línea de regresión
st.subheader("Diagrama de dispersión con regresión")
fig, ax = plt.subplots(figsize=(6, 6))
sns.regplot(
    x=df_f[metodos[0]],
    y=df_f[metodos[1]],
    ax=ax,
    scatter_kws={"alpha": 0.7},
    line_kws={"color": "red"}
)

ax.set_xlabel(metodos[0])
ax.set_ylabel(metodos[1])
ax.set_title(f"Comparación entre métodos\nCorrelación: {corr:.3f}", fontsize=12)

# Subtítulo con rango de profundidad
ax.text(
    0.5, -0.15,
    f"Rango de profundidad: {tope:.2f} - {base:.2f} ft",
    transform=ax.transAxes,
    ha="center", va="center",
    fontsize=10, color="gray"
)

st.pyplot(fig)

# Guardar gráfico como .jpeg
filename = f"Comparacion_{metodos[0]}_vs_{metodos[1]}_{tope:.0f}-{base:.0f}ft.jpeg"
fig.savefig(filename, format="jpeg", dpi=300, bbox_inches="tight")

with open(filename, "rb") as f:
    st.download_button("💾 Descargar gráfico (.jpeg)", f, file_name=filename, mime="image/jpeg")
