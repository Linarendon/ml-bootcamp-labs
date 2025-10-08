# app_mejores_parejas.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ranking mejores parejas", layout="wide")

st.title("🔎 Ranking de las mejores parejas (Curva Real vs Curva de Inversión)")

# -----------------------------
# Subir archivo CSV
# -----------------------------
uploaded = st.file_uploader("Sube el archivo CSV (separado por ';')", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, sep=";")
else:
    st.info("Sube tu archivo CSV para comenzar.")
    st.stop()

# Normalizar nombres de columnas
df.columns = [c.strip().upper() for c in df.columns]

# Validaciones
for col in ["POZO", "CURVA_REGISTRO", "CURVA_INVERSION", "CORRELACION", "CONSISTENCIA"]:
    if col not in df.columns:
        st.error(f"Falta la columna requerida: {col}")
        st.stop()

# -----------------------------
# Filtro por INTERVALO (si existe)
# -----------------------------
if "INTERVALO" in df.columns:
    intervalos = sorted(df["INTERVALO"].dropna().unique())
    seleccion = st.multiselect("Selecciona uno o más intervalos a analizar:", intervalos, default=intervalos)
    df = df[df["INTERVALO"].isin(seleccion)]
else:
    st.info("No se encontró la columna 'INTERVALO', se usarán todos los datos disponibles.")

# -----------------------------
# Crear correlación absoluta
# -----------------------------
df["CORRELACION_ABS"] = df["CORRELACION"].abs()

# Convertir consistencia a 1/0
import unicodedata

def parse_consistency(val):
    """Devuelve 1 si la cadena indica consistente (si/yes/consistente/1/true...), 0 si indica no.
       Normaliza acentos, espacios y mayúsculas."""
    if pd.isna(val):
        return 0
    s = str(val)
    # Normalizar acentos y pasar a minúsculas
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    s = s.strip().lower()

    # valores que consideramos True
    true_set = {'si','sí','s','yes','y','1','true','t','consistente','consistent','consist'}
    # valores que consideramos False
    false_set = {'no','n','0','false','f','inconsistente','inconsistent','not consistent'}

    if s in true_set:
        return 1
    if s in false_set:
        return 0
    # Revisar subcadenas (por si viene 'INCONSISTENTE' o 'CONSISTENTE' con texto extra)
    if 'inconsist' in s:
        return 0
    if 'consist' in s:
        return 1
    # Intentar interpretar numéricamente
    try:
        n = float(s)
        return 1 if n != 0 else 0
    except:
        # fallback: marcar como no consistente (0)
        return 0

# aplicar al DataFrame
df["CONSIST_FLAG"] = df["CONSISTENCIA"].apply(parse_consistency)


# -----------------------------
# Resumen por pareja
# -----------------------------
summary = df.groupby(["CURVA_REGISTRO", "CURVA_INVERSION"]).agg(
    mean_abs=("CORRELACION_ABS", "mean"),
    median_abs=("CORRELACION_ABS", "median"),
    std_abs=("CORRELACION_ABS", "std"),
    n=("CORRELACION_ABS", "count"),
    consist_pct=("CONSIST_FLAG", "mean")
).reset_index()

summary["consist_pct"] = (summary["consist_pct"] * 100).round(1)

# Score: media * (1 - std) * consistencia
summary["score"] = summary["mean_abs"] * (1 - summary["std_abs"].fillna(0)) * (summary["consist_pct"]/100)

# Ordenar por score
summary = summary.sort_values("score", ascending=False).reset_index(drop=True)

st.subheader("📊 Ranking de parejas (ordenadas por score)")
if summary.empty:
    st.warning("No hay datos consistentes para los intervalos seleccionados.")
    st.stop()
else:
    st.dataframe(summary)

# Función auxiliar para exportar figuras
def fig_to_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# -----------------------------
# Heatmap (promedios por pareja)
# -----------------------------
if not summary.empty:
    st.subheader("🔥 Heatmap de correlación absoluta promedio")
    pivot = summary.pivot(index="CURVA_REGISTRO", columns="CURVA_INVERSION", values="mean_abs")

    if pivot.empty:
        st.warning("No hay suficientes datos para generar el heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Correlación absoluta promedio'})
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)

        # Botón para descargar heatmap
        st.download_button(
            label="💾 Descargar heatmap",
            data=fig_to_bytes(fig, fmt="png"),
            file_name="heatmap_correlacion.png",
            mime="image/png"
        )

# -----------------------------
# Boxplot detallado
# -----------------------------
if not df.empty:
    st.subheader("📦 Distribución de correlaciones absolutas por pareja")

    fig2, ax2 = plt.subplots(figsize=(12,6))
    sns.boxplot(data=df, x="CURVA_REGISTRO", y="CORRELACION_ABS", hue="CURVA_INVERSION", ax=ax2)
    ax2.set_title("Distribución de correlaciones absolutas")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Botón para descargar boxplot
    st.download_button(
        label="💾 Descargar boxplot",
        data=fig_to_bytes(fig2, fmt="png"),
        file_name="boxplot_correlacion.png",
        mime="image/png"
    )

# -----------------------------
# Conclusión automática
# -----------------------------
if not summary.empty:
    best = summary.iloc[0]
    st.success(
        f"✅ La mejor pareja es **{best['CURVA_REGISTRO']} – {best['CURVA_INVERSION']}** "
        f"con promedio {best['mean_abs']:.2f}, mediana {best['median_abs']:.2f}, "
        f"consistencia {best['consist_pct']}% y score {best['score']:.2f}."
    )
