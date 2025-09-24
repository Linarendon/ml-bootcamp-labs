# MATRIZ_CONFUSION_V3.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# 1. Cargar datos
# -----------------------------
FILE_PATH = "/workspaces/ml-bootcamp-labs/data/raw/LLANITO_LOGS_DISCRETOS_cvs.csv"

st.title("📊 Comparación de curvas discretas con Matriz de Confusión y Métricas")

@st.cache_data
def load_data():
    df = pd.read_csv(FILE_PATH, sep=";")
    df.columns = df.columns.str.strip()  # limpiar espacios
    return df

df = load_data()
df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce").astype(int)

# -----------------------------
# 2. Selección de opciones
# -----------------------------
pozos = df["POZO"].unique().tolist()
pozo = st.selectbox("Seleccione el pozo", pozos)

curvas_reales = ["RT3_PERM", "RT3_POR", "RT3_VSH"]
curvas_modelo = ["RT3_XPLT", "RT3_IP", "RT3_VPVS"]

curva_real = st.selectbox("Seleccione la curva REAL", curvas_reales)
curva_modelo = st.selectbox("Seleccione la curva del MODELO", curvas_modelo)

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
# 4. Matriz de confusión y métricas
# -----------------------------
if curva_real not in df_pozo.columns or curva_modelo not in df_pozo.columns:
    st.error(f"⚠️ La curva '{curva_real}' o '{curva_modelo}' no existe en los datos.")
else:
    if df_pozo.empty:
        st.warning("⚠️ No hay datos en el intervalo seleccionado.")
    else:
        y_true = df_pozo[curva_real].astype(str)
        y_pred = df_pozo[curva_modelo].astype(str)

        labels = np.unique(list(y_true) + list(y_pred))

        if len(labels) == 0:
            st.warning("⚠️ No se encontraron valores válidos.")
        else:
            # 📌 Matriz de confusión absoluta
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicción (Modelo)")
            ax.set_ylabel("Real")
            ax.set_title(f"Matriz de confusión absoluta: {curva_real} vs {curva_modelo}")
            st.pyplot(fig)

            # 📌 Matriz de confusión en porcentajes
            if cm.sum() > 0:
                cm_percent = cm.astype("float") / cm.sum() * 100
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Oranges",
                            xticklabels=labels, yticklabels=labels, ax=ax2)
                ax2.set_xlabel("Predicción (Modelo)")
                ax2.set_ylabel("Real")
                ax2.set_title(f"Matriz de confusión (% del total)")
                st.pyplot(fig2)

            # 📌 Métricas globales
            total_puntos = cm.sum()
            aciertos = np.trace(cm)
            accuracy_global = aciertos / total_puntos if total_puntos > 0 else 0

            st.subheader("📊 Métricas Globales")
            st.write(f"🔹 Total de puntos evaluados: **{total_puntos}**")
            st.write(f"🔹 Total de aciertos: **{aciertos}**")
            st.write(f"🔹 Accuracy global: **{accuracy_global:.2%}**")

            # 📌 Métricas por clase
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            st.subheader("📈 Métricas de Clasificación por Clase")
            st.write(pd.DataFrame(report).transpose())
