# MATRIZ_CONFUSION_V1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# 1. Cargar datos
# -----------------------------
FILE_PATH = "/workspaces/ml-bootcamp-labs/data/raw/LLANITO_LOGS_DISCRETOS_cvs.csv"

st.title("📊 Comparación de curvas discretas con Matriz de Confusión")

@st.cache_data
def load_data():
    df = pd.read_csv(FILE_PATH, sep=";")
    df.columns = df.columns.str.strip()  # limpiar espacios
    return df

df = load_data()

# Asegurar que DEPTH sea numérico e int
df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce").astype(int)


# -----------------------------
# 2. Selección de opciones
# -----------------------------
pozos = df["POZO"].unique().tolist()
pozo = st.selectbox("Seleccione el pozo", pozos)

# ✅ Curvas reales y de modelo disponibles
curvas_reales = ["RT3_PERM", "RT3_POR", "RT3_VSH"]
curvas_modelo = ["RT3_XPLT", "RT3_IP", "RT3_VPVS"]

curva_real = st.selectbox("Seleccione la curva REAL", curvas_reales)
curva_modelo = st.selectbox("Seleccione la curva del MODELO", curvas_modelo)

# Intervalo de profundidad
col1, col2 = st.columns(2)
with col1:
    depth_min = st.number_input("Profundidad mínima", value=float(df["DEPTH"].min()))
with col2:
    depth_max = st.number_input("Profundidad máxima", value=float(df["DEPTH"].max()))

# -----------------------------
# 3. Filtrado de datos
# -----------------------------
df_pozo = df[(df["POZO"] == pozo) &
             (df["DEPTH"] >= depth_min) &
             (df["DEPTH"] <= depth_max)].copy()

st.write(f"🔎 Filtrado: {len(df_pozo)} filas seleccionadas")

# -----------------------------
# 4. Matriz de confusión
# -----------------------------
if curva_real not in df_pozo.columns or curva_modelo not in df_pozo.columns:
    st.error(f"⚠️ La curva '{curva_real}' o '{curva_modelo}' no existe en los datos.")
    st.write("Columnas disponibles:", df_pozo.columns.tolist())
else:
    if df_pozo.empty:
        st.warning("⚠️ No hay datos en el intervalo seleccionado. Ajusta el rango de profundidad o el pozo.")
    else:
        y_true = df_pozo[curva_real].astype(str)
        y_pred = df_pozo[curva_modelo].astype(str)

        labels = np.unique(list(y_true) + list(y_pred))

        if len(labels) == 0:
            st.warning("⚠️ No se encontraron valores válidos para generar la matriz de confusión.")
        else:
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
            plt.title(f"Matriz de confusión: {curva_real} vs {curva_modelo}")
            st.pyplot(fig)
