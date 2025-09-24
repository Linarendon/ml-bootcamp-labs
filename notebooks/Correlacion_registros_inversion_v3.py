import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

# =====================
# 1. Configuración
# =====================
st.set_page_config(page_title="Comparación IP vs Vp/Vs", layout="wide")

# =====================
# 2. Cargar datos
# =====================
ruta = "/workspaces/ml-bootcamp-labs/data/raw/Correlacion_Registros_Inversion_cvs.csv"

@st.cache_data
def cargar_datos():
    df = pd.read_csv(ruta, sep=";")
    return df

df = cargar_datos()

st.title("📊 Comparación de correlaciones: IP vs Vp/Vs")
st.markdown("Este análisis compara la calidad de correlación entre curvas de **inversión sísmica** "
            "(`IP` y `Vp/Vs`) respecto a curvas de pozo.")

# =====================
# 3. Filtros
# =====================
df = df[df["CURVA_INVERSION"].isin(["IP", "VPVS"])]

curva_registro_sel = st.multiselect(
    "Selecciona las curvas de registro a analizar:",
    options=df["CURVA_REGISTRO"].unique(),
    default=list(df["CURVA_REGISTRO"].unique())
)

df = df[df["CURVA_REGISTRO"].isin(curva_registro_sel)]

# =====================
# 4. Estadísticas descriptivas
# =====================
stats = df.groupby("CURVA_INVERSION")["CORRELACION"].agg(
    ["mean", "median", "std", "count"]
).reset_index()

st.subheader("📑 Resumen estadístico")
st.dataframe(stats, use_container_width=True)

# =====================
# 5. Test estadístico
# =====================
ip_corr = df[df["CURVA_INVERSION"]=="IP"]["CORRELACION"]
vpvs_corr = df[df["CURVA_INVERSION"]=="VPVS"]["CORRELACION"]

t_stat, p_val = ttest_ind(ip_corr, vpvs_corr, equal_var=False)
u_stat, p_u = mannwhitneyu(ip_corr, vpvs_corr, alternative="two-sided")

st.subheader("📌 Pruebas estadísticas")
st.markdown(f"""
- **T-test (IP vs Vp/Vs)**: t = `{t_stat:.3f}`, p = `{p_val:.3f}`  
- **Mann–Whitney U (IP vs Vp/Vs)**: U = `{u_stat:.3f}`, p = `{p_u:.3f}`  
""")

if p_val < 0.05 or p_u < 0.05:
    st.success("✅ Hay diferencia estadísticamente significativa entre IP y Vp/Vs.")
else:
    st.info("ℹ️ No se encontró diferencia significativa entre IP y Vp/Vs.")

# =====================
# 6. Visualización
# =====================
st.subheader("📊 Distribución de correlaciones")

fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=df, x="CURVA_INVERSION", y="CORRELACION", palette="Set2", ax=ax)
sns.swarmplot(data=df, x="CURVA_INVERSION", y="CORRELACION", color=".25", ax=ax)
ax.set_title("Comparación de correlaciones: IP vs Vp/Vs")
st.pyplot(fig)

# =====================
# 7. Detalle por curva de registro
# =====================
st.subheader("🔍 Detalle por curva de registro")

fig2, ax2 = plt.subplots(figsize=(10,6))
sns.boxplot(data=df, x="CURVA_REGISTRO", y="CORRELACION",
            hue="CURVA_INVERSION", palette="Set2", ax=ax2)
ax2.set_title("Correlaciones por curva de registro")
st.pyplot(fig2)

