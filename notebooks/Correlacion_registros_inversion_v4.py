# app_correlacion_abs.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Comparación de Correlaciones Absolutas", layout="wide")

st.title("📊 Comparación de correlaciones absolutas entre IP y Vp/Vs")

# -----------------------------
# Subir archivo CSV
# -----------------------------
uploaded = st.file_uploader("Sube el archivo CSV (separado por ';')", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, sep=";")
else:
    st.info("Sube tu archivo CSV para comenzar.")
    st.stop()

# Normalizar columnas
df.columns = [c.strip().upper() for c in df.columns]

# Verificar que exista la columna CORRELACION
if "CORRELACION" not in df.columns:
    st.error("El archivo debe contener la columna 'CORRELACION'")
    st.stop()

# Crear columna de correlación absoluta
df["CORRELACION_ABS"] = df["CORRELACION"].abs()

# -----------------------------
# Filtros
# -----------------------------
pozos = sorted(df["POZO"].unique())
curvas_reg = sorted(df["CURVA_REGISTRO"].unique())

pozo_sel = st.sidebar.multiselect("Filtrar por pozos", pozos, default=pozos)
curva_sel = st.sidebar.multiselect("Filtrar por curvas de registro", curvas_reg, default=curvas_reg)

df_f = df[(df["POZO"].isin(pozo_sel)) & (df["CURVA_REGISTRO"].isin(curva_sel))].copy()

if df_f.empty:
    st.warning("⚠️ No hay datos para los filtros seleccionados.")
    st.stop()

# ----------------------------------
# Filtrar solo correlaciones consistentes
df_f = df_f[df_f["CONSISTENCIA"].str.upper() == "CONSISTENTE"]

# -----------------------------
# Resumen estadístico
# -----------------------------
st.subheader("Resumen estadístico por curva de inversión")
resumen = df_f.groupby("CURVA_INVERSION")["CORRELACION_ABS"].describe().T
st.dataframe(resumen)

# -----------------------------
# Gráfico 1: Boxplot comparando IP vs Vp/Vs
# -----------------------------
st.subheader("Distribución de correlaciones absolutas (IP vs Vp/Vs)")

fig1, ax1 = plt.subplots(figsize=(8,6))
sns.boxplot(data=df_f, x="CURVA_INVERSION", y="CORRELACION_ABS", ax=ax1, palette="Set2")
sns.stripplot(data=df_f, x="CURVA_INVERSION", y="CORRELACION_ABS", ax=ax1, color="black", alpha=0.5)
ax1.set_title("Comparación de correlaciones absolutas")
st.pyplot(fig1)

# Botón para descargar figura 1
buf1 = io.BytesIO()
fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    label="💾 Descargar Gráfico 1 (PNG)",
    data=buf1.getvalue(),
    file_name="correlacion_abs_ip_vs_vpvs.png",
    mime="image/png"
)

# -----------------------------
# Gráfico 2: Boxplot por curva de registro
# -----------------------------
st.subheader("Correlaciones absolutas por curva de registro")

fig2, ax2 = plt.subplots(figsize=(12,6))
sns.boxplot(data=df_f, x="CURVA_REGISTRO", y="CORRELACION_ABS", hue="CURVA_INVERSION", ax=ax2)
ax2.set_title("Correlaciones absolutas por curva de registro")
st.pyplot(fig2)

# Botón para descargar figura 2
buf2 = io.BytesIO()
fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    label="💾 Descargar Gráfico 2 (PNG)",
    data=buf2.getvalue(),
    file_name="correlacion_abs_por_curva.png",
    mime="image/png"
)

# -----------------------------
# Conclusión simple
# -----------------------------
mean_scores = df_f.groupby("CURVA_INVERSION")["CORRELACION_ABS"].mean()
mejor = mean_scores.idxmax()
st.success(f"📌 En promedio, la curva de inversión con mejor correlación absoluta es **{mejor}** "
           f"({mean_scores[mejor]:.2f} en promedio).")
