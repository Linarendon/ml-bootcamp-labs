import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ==========================
# Configuración inicial
# ==========================
st.title("📊 Análisis de Curvas de Roca")
st.write("Comparación entre curvas reales y predichas con matriz de confusión")

# Cargar archivo
@st.cache_data
def cargar_datos():
    ruta = "/workspaces/ml-bootcamp-labs/data/raw/CURVAS_CLASIFICADAS_CVS.csv"
    return pd.read_csv(ruta, sep=";")

df = cargar_datos()

# Asegurar que Profundidad es numérica
df["Profundidad"] = pd.to_numeric(df["Profundidad"], errors="coerce")

# ==========================
# Selección de parámetros
# ==========================
st.sidebar.header("⚙️ Parámetros de análisis")

# Rango de profundidades
min_prof = int(df["Profundidad"].min())
max_prof = int(df["Profundidad"].max())

rango_slider = st.sidebar.slider(
    "Selecciona rango de profundidad",
    min_value=min_prof, max_value=max_prof,
    value=(min_prof, min_prof + 100),
    step=1
)

prof_min_input = st.sidebar.number_input("Profundidad mínima", min_value=min_prof, max_value=max_prof, value=rango_slider[0])
prof_max_input = st.sidebar.number_input("Profundidad máxima", min_value=min_prof, max_value=max_prof, value=rango_slider[1])

prof_min = min(prof_min_input, rango_slider[0])
prof_max = max(prof_max_input, rango_slider[1])

# Selección de curvas
curvas = df.columns.drop(["Profundidad", "Pozo"])
curva_real = st.sidebar.selectbox("Curva real", curvas)
curva_pred = st.sidebar.selectbox("Curva predicha", curvas)

# Tolerancia de alineación
tolerancia = st.sidebar.number_input("Tolerancia de alineación (ft)", min_value=0, max_value=50, value=10)

# ==========================
# Filtrado y alineación
# ==========================
df_filtrado = df[(df["Profundidad"] >= prof_min) & (df["Profundidad"] <= prof_max)]

if df_filtrado.empty:
    st.warning("⚠️ No hay datos en este rango de profundidad.")
else:
    # Alinear curvas con tolerancia
    df_sorted = df_filtrado.sort_values("Profundidad")
    df_real = df_sorted[["Profundidad", curva_real]].dropna().rename(columns={curva_real: "real"})
    df_pred = df_sorted[["Profundidad", curva_pred]].dropna().rename(columns={curva_pred: "predicha"})

    df_alineado = pd.merge_asof(
        df_real, df_pred,
        on="Profundidad",
        direction="nearest",
        tolerance=tolerancia
    ).dropna()

    if df_alineado.empty:
        st.warning("❌ No hay datos alineados entre la curva real y la predicha en este rango.")
    else:
        # ==========================
        # Matriz de confusión
        # ==========================
        cm = confusion_matrix(df_alineado["real"], df_alineado["predicha"])
        labels = np.unique(list(df_alineado["real"].unique()) + list(df_alineado["predicha"].unique()))

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de Confusión\nProfundidad {prof_min}-{prof_max} ft")
        st.pyplot(fig)

        # Normalizada (%)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión Normalizada (%)")
        st.pyplot(fig)

        # ==========================
        # Métricas
        # ==========================
        st.subheader("📌 Métricas de Clasificación")
        st.text(classification_report(df_alineado["real"], df_alineado["predicha"]))

        # Total de puntos evaluados
        total = len(df_alineado)
        st.write(f"📊 Total de puntos evaluados: **{total}**")

