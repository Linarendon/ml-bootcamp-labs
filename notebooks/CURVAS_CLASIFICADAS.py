import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================
# 1. Cargar datos
# ============================
FILE_PATH = "/workspaces/ml-bootcamp-labs/data/raw/CURVAS_CLASIFICADAS_CVS.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(FILE_PATH, sep=";")
    return df

df = load_data()

# ============================
# 2. Título
# ============================
st.title("📊 Validación de curvas clasificadas con matriz de confusión")

st.write("Datos cargados:")
st.dataframe(df.head())

# ============================
# 3. Selección de rango de profundidad
# ============================
min_prof = int(df["Profundidad"].min())
max_prof = int(df["Profundidad"].max())

st.sidebar.header("⚙️ Configuración")

prof_min = st.sidebar.number_input("Profundidad mínima", min_value=min_prof, max_value=max_prof, value=min_prof)
prof_max = st.sidebar.number_input("Profundidad máxima", min_value=min_prof, max_value=max_prof, value=max_prof)

# ============================
# 4. Selección de curvas
# ============================
curvas = df["Tipo"].unique().tolist()
st.sidebar.subheader("Seleccionar curvas para validar")

curva_real = st.sidebar.selectbox("Curva real (ground truth)", ["RT3_K","RT3_POR","RT3_VSH"])
curva_pred = st.sidebar.selectbox("Curva predicha (modelo)", ["RT3_VPVS","RT3_IP","RT3_XPLT"])

# ============================
# 5. Filtrar datos
# ============================
df_intervalo = df[(df["Profundidad"] >= prof_min) & (df["Profundidad"] <= prof_max)]

# Pivotear para tener columnas por curva
df_pivot = df_intervalo.pivot_table(index=["Profundidad","Pozo"], columns="Tipo", values="Valor", aggfunc="first").reset_index()

if curva_real not in df_pivot.columns or curva_pred not in df_pivot.columns:
    st.error(f"❌ No hay datos para {curva_real} o {curva_pred} en el intervalo seleccionado.")
else:
    # ============================
    # 6. Preparar vectores
    # ============================
    datos = df_pivot.dropna(subset=[curva_real, curva_pred])
    y_true = datos[curva_real].astype(int)
    y_pred = datos[curva_pred].astype(int)

    if y_true.empty or y_pred.empty:
        st.error("⚠️ No hay datos suficientes después del filtrado.")
    else:
        # ============================
        # 7. Matriz de confusión
        # ============================
        labels = [1, 2, 3]  # tus clases
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Normalizada (%)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot matriz absoluta
        fig, ax = plt.subplots(1,2, figsize=(14,6))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax[0])
        ax[0].set_xlabel("Predicho")
        ax[0].set_ylabel("Real")
        ax[0].set_title("Matriz de Confusión (conteos)")

        sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax[1])
        ax[1].set_xlabel("Predicho")
        ax[1].set_ylabel("Real")
        ax[1].set_title("Matriz de Confusión (%)")

        st.pyplot(fig)

        # ============================
        # 8. Métricas globales
        # ============================
        total = np.sum(cm)
        aciertos = np.trace(cm)
        accuracy = aciertos / total * 100

        st.subheader("📈 Métricas de validación")
        st.write(f"✅ Total de puntos evaluados: {total}")
        st.write(f"🎯 Aciertos (diagonal): {aciertos}")
        st.write(f"📊 Accuracy global: {accuracy:.2f}%")

        # ============================
        # 9. Reporte por clase
        # ============================
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        st.subheader("📑 Reporte de clasificación por clase")
        st.dataframe(pd.DataFrame(report).transpose())
