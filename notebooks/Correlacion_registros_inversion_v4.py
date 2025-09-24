import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro

# ======================
# Configuración
# ======================
st.set_page_config(page_title="Análisis de correlaciones IP vs Vp/Vs", layout="wide")

st.title("📊 Análisis de correlaciones: IP vs Vp/Vs")
st.markdown("Esta aplicación permite comparar las correlaciones entre curvas de registros e inversión sísmica (IP y Vp/Vs).")

# ======================
# 1. Subida de archivo
# ======================
uploaded_file = st.file_uploader("📂 Cargar archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # ======================
    # 2. Análisis estadístico
    # ======================
    resultados = []

    for curva in df["CURVA_REGISTRO"].unique():
        df_curva = df[df["CURVA_REGISTRO"] == curva]

        ip = df_curva[df_curva["CURVA_INVERSION"] == "IP"]["CORRELACION"].dropna()
        vpvs = df_curva[df_curva["CURVA_INVERSION"] == "VPVS"]["CORRELACION"].dropna()

        n = min(len(ip), len(vpvs))
        ip = ip.iloc[:n]
        vpvs = vpvs.iloc[:n]

        # Test de normalidad
        p_shapiro_ip = shapiro(ip)[1] if len(ip) > 3 else 1
        p_shapiro_vpvs = shapiro(vpvs)[1] if len(vpvs) > 3 else 1

        if p_shapiro_ip > 0.05 and p_shapiro_vpvs > 0.05:
            stat, p_value = ttest_rel(ip, vpvs)
            test = "t-test pareado"
        else:
            try:
                stat, p_value = wilcoxon(ip, vpvs)
                test = "Wilcoxon"
            except ValueError:
                p_value = 1
                test = "Wilcoxon (no válido)"

        resultados.append({
            "Curva_registro": curva,
            "Media_IP": ip.mean(),
            "Media_VPVS": vpvs.mean(),
            "Test_usado": test,
            "p_value": p_value,
            "Conclusión": "Diferencia significativa" if p_value < 0.05 else "No significativa"
        })

    df_resultados = pd.DataFrame(resultados)

    # ======================
    # 3. Mostrar resultados
    # ======================
    st.subheader("📑 Resultados estadísticos")
    st.dataframe(df_resultados)

    # ======================
    # 4. Visualización
    # ======================
    st.subheader("📈 Boxplots comparativos")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df, x="CURVA_REGISTRO", y="CORRELACION", hue="CURVA_INVERSION", ax=ax)
    ax.set_title("Comparación de correlaciones entre IP y Vp/Vs por curva de registro")
    ax.set_xlabel("Curva de registro")
    ax.set_ylabel("Correlación")
    st.pyplot(fig)

else:
    st.info("👆 Sube un archivo CSV para comenzar el análisis.")
