# ==========================================
# Análisis Interactivo de Desfase Pozo–Sísmica por Campo (Versión mejorada)
# ==========================================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# ======================
# Configuración inicial
# ======================
st.set_page_config(page_title="Análisis de Desfases Pozo–Sísmica", layout="wide")
st.title("🛢️ Análisis de Desfases Pozo–Sísmica por Campo")
st.markdown("""
Esta aplicación permite analizar los **desfases entre Well Pick y Surface Pick**
y agrupar los resultados **por campos petroleros** para identificar patrones estructurales o de calidad de ajuste.
""")

# ======================
# 1. Carga del archivo
# ======================
st.sidebar.header("📂 Carga de Datos")

uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Detectar tipo de archivo
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success(f"Archivo cargado: **{uploaded_file.name}**")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()

    # ======================
    # 2. Selección de columnas clave
    # ======================
    st.sidebar.subheader("🧩 Selecciona columnas relevantes")
    col_field = st.sidebar.selectbox("Columna de campo:", df.columns)
    col_well = st.sidebar.selectbox("Columna de pozo:", df.columns)
    col_x = st.sidebar.selectbox("Columna X (coordenada este):", df.columns)
    col_y = st.sidebar.selectbox("Columna Y (coordenada norte):", df.columns)
    col_pickz = st.sidebar.selectbox("Columna de profundidad (Pick Z):", df.columns)
    col_delta = st.sidebar.selectbox("Columna Delta (Shift):", df.columns)
    col_absdelta = st.sidebar.selectbox("Columna Absolute Delta:", df.columns)

    # ======================
    # 3. Filtro por campo
    # ======================
    campos = df[col_field].unique().tolist()
    campos_selected = st.multiselect(
        "Selecciona los campos a analizar:",
        sorted(campos),
        default=campos[:3]
    )

    df_sel = df[df[col_field].isin(campos_selected)]

    st.markdown(f"**Número de pozos seleccionados:** {df_sel[col_well].nunique()}")

    # 🎨 Paleta de colores fija
    palette = sns.color_palette("tab10", len(campos_selected))
    color_dict = dict(zip(campos_selected, palette))

    # ======================
    # 4. Estadísticas descriptivas
    # ======================
    st.subheader("📈 Estadísticas descriptivas")
    stats = df_sel[[col_delta, col_absdelta]].describe(percentiles=[0.25, 0.5, 0.75])
    st.dataframe(stats)

    # Descargar estadísticas
    st.download_button(
        label="📥 Descargar estadísticas (CSV)",
        data=stats.to_csv().encode("utf-8"),
        file_name="estadisticas_shifts_por_campo.csv",
        mime="text/csv"
    )

    # Función auxiliar para descargar imágenes
    def download_plot(fig, filename_base):
        buffer_png = BytesIO()
        buffer_jpg = BytesIO()
        fig.savefig(buffer_png, format="png", dpi=300, bbox_inches="tight")
        fig.savefig(buffer_jpg, format="jpeg", dpi=300, bbox_inches="tight")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"⬇️ Descargar {filename_base}.png",
                data=buffer_png.getvalue(),
                file_name=f"{filename_base}.png",
                mime="image/png"
            )
        with col2:
            st.download_button(
                label=f"⬇️ Descargar {filename_base}.jpeg",
                data=buffer_jpg.getvalue(),
                file_name=f"{filename_base}.jpeg",
                mime="image/jpeg"
            )

    # ======================
    # 5. Histograma
    # ======================
    st.subheader("📊 Distribución de Delta (por campo)")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df_sel, x=col_delta, hue=col_field, bins=30, kde=True, ax=ax1, palette=color_dict)
    ax1.set_title("Distribución de Delta (Shift) por campo")
    ax1.set_xlabel("Delta (ft o ms)")
    ax1.set_ylabel("Frecuencia")
    st.pyplot(fig1)
    download_plot(fig1, "Histograma_Delta_por_Campo")

    # ======================
    # 6. Boxplot
    # ======================
    st.subheader("🧭 Boxplot de Delta por campo")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=col_field, y=col_delta, data=df_sel, palette=color_dict, ax=ax2)
    plt.xticks(rotation=45)
    ax2.set_title("Distribución de Delta por campo")
    st.pyplot(fig2)
    download_plot(fig2, "Boxplot_Delta_por_Campo")

    # ======================
    # 7. Mapa espacial
    # ======================
    st.subheader("🗺️ Mapa espacial de Delta por campo")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sc = ax3.scatter(
        df_sel[col_x], df_sel[col_y],
        c=df_sel[col_delta],
        cmap="RdBu_r", s=60, edgecolor="k"
    )
    plt.colorbar(sc, label="Delta (Shift)")
    ax3.set_xlabel("X (Este)")
    ax3.set_ylabel("Y (Norte)")
    ax3.set_title("Distribución espacial de Delta")
    st.pyplot(fig3)
    download_plot(fig3, "Mapa_Espacial_Delta")

    # ======================
    # 8. Correlación Delta vs Profundidad
    # ======================
    st.subheader("📉 Correlación Delta vs Profundidad (Pick Z)")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_sel,
        x=col_pickz, y=col_delta,
        hue=col_field,
        palette=color_dict,
        ax=ax4
    )
    ax4.axhline(0, color="red", linestyle="--")
    ax4.set_xlabel("Profundidad (Pick Z)")
    ax4.set_ylabel("Delta (Shift)")
    ax4.set_title("Correlación Delta vs Profundidad por campo")
    st.pyplot(fig4)
    download_plot(fig4, "Correlacion_Delta_vs_Profundidad")

    # ======================
    # 9. Heatmap de correlaciones
    # ======================
    if st.checkbox("🔍 Mostrar matriz de correlaciones numéricas"):
        num_cols = df_sel.select_dtypes(include="number")
        corr = num_cols.corr()
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
        st.pyplot(fig5)
        download_plot(fig5, "Matriz_Correlaciones")

else:
    st.warning("👈 Carga un archivo CSV o Excel para comenzar el análisis.")
