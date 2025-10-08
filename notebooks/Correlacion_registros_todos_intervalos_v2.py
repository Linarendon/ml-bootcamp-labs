# Correlacion_heatmap_consistencia_app_v3.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Heatmap Correlaciones (Con Consistencia)", layout="wide")

# ==========================================================
# 1️⃣ Funciones auxiliares
# ==========================================================
def str_to_bool_consistent(x):
    """Convierte valores como 'si', 'no', 'true', 'false' a booleanos."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("si", "sí", "yes", "y", "1", "true", "t", "ok"):
        return True
    if s in ("no", "n", "0", "false", "f"):
        return False
    return np.nan


def expected_sign(reg, inv):
    """Reglas esperadas (devuelve 'Positiva' / 'Negativa' / 'N/A')."""
    r = str(reg).strip().upper()
    i = str(inv).strip().upper()
    if r == "GR" and i in ("IP", "VPVS", "XPLT"):
        return "Positiva"
    if r == "VSH" and i in ("IP", "VPVS", "XPLT"):
        return "Positiva"
    if r in ("PERM", "PERMEABILIDAD") and i in ("IP", "VPVS", "XPLT"):
        return "Negativa"
    if r in ("POR", "POROSIDAD", "POROSITY") and i in ("IP", "VPVS", "XPLT"):
        return "Negativa"
    if r.startswith("GR"):
        return "Positiva"
    if r.startswith("VSH"):
        return "Positiva"
    if r.startswith("PER") or r.startswith("K"):
        return "Negativa"
    if r.startswith("POR"):
        return "Negativa"
    return "N/A"


def build_annot_string(corr_val, cons_flag, exp_sign, style="num_sym"):
    """Texto que aparece dentro de cada celda del heatmap."""
    if pd.isna(corr_val):
        return ""
    num = f"{corr_val:.2f}"
    symbol = "✅" if cons_flag else "❌"
    if style == "num":
        return num
    if style == "num_sym":
        return f"{num}\n{symbol}"
    if style == "num_sym_rule":
        return f"{num}\n{symbol}\n({exp_sign})"
    return num

# ==========================================================
# 2️⃣ Interfaz principal
# ==========================================================
st.title("🔥 Heatmap de Correlaciones (Registros vs Inversión) con Consistencia")

uploaded = st.sidebar.file_uploader("📁 Sube tu archivo CSV (separador ';')", type=["csv"])
ruta_default = "/workspaces/ml-bootcamp-labs/data/raw/Correlacion_Registros_Inversion_XP_cvs.csv"

if uploaded is not None:
    df = pd.read_csv(uploaded, sep=";")
else:
    df = pd.read_csv(ruta_default, sep=";")

# Limpieza básica
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=["POZO", "CURVA_REGISTRO", "CURVA_INVERSION", "CORRELACION"], how="any")

st.sidebar.markdown("**Columnas detectadas:**")
st.sidebar.write(list(df.columns))

st.subheader("Vista previa de los datos")
st.dataframe(df.head(10))

# ==========================================================
# 3️⃣ Selección de intervalo
# ==========================================================
if "INTERVALO" in df.columns:
    intervalos = sorted(df["INTERVALO"].dropna().unique().tolist())
    intervalo_sel = st.sidebar.selectbox("🪨 Selecciona intervalo", intervalos)
    df = df[df["INTERVALO"] == intervalo_sel].copy()
    st.sidebar.success(f"Intervalo seleccionado: {intervalo_sel}")
else:
    st.sidebar.warning("⚠️ No hay columna 'INTERVALO' en el archivo.")

# ==========================================================
# 4️⃣ Preparación de datos
# ==========================================================
df["CONSISTENCIA_BOOL"] = df["CONSISTENCIA"].apply(str_to_bool_consistent)
df["PAR_CURVAS"] = df["CURVA_REGISTRO"].str.upper() + "_vs_" + df["CURVA_INVERSION"].str.upper()

for col in ["POZO", "CURVA_REGISTRO", "CURVA_INVERSION"]:
    df[col] = df[col].astype(str).str.strip().str.upper()

# ==========================================================
# 5️⃣ Filtros laterales
# ==========================================================
st.sidebar.header("Filtros")

pozos = sorted(df["POZO"].unique().tolist())
pozo_sel = st.sidebar.multiselect("Pozos", pozos, default=pozos)

curvas_reg = sorted(df["CURVA_REGISTRO"].unique().tolist())
curvas_inv = sorted(df["CURVA_INVERSION"].unique().tolist())

cur_reg_sel = st.sidebar.multiselect("Curvas de Registro", curvas_reg, default=curvas_reg)
cur_inv_sel = st.sidebar.multiselect("Curvas de Inversión", curvas_inv, default=curvas_inv)

mode = st.sidebar.radio("Modo de visualización", options=["Solo correlación", "Correlación + Consistencia y Regla"])
annot_style = st.sidebar.selectbox(
    "Estilo de anotación",
    options=[("Número", "num"), ("Número + símbolo (✅/❌)", "num_sym"), ("Número + símbolo + regla esperada", "num_sym_rule")],
    index=1
)[1]

# ==========================================================
# 6️⃣ Filtrado según selección
# ==========================================================
df_f = df[
    (df["POZO"].isin(pozo_sel)) &
    (df["CURVA_REGISTRO"].isin(cur_reg_sel)) &
    (df["CURVA_INVERSION"].isin(cur_inv_sel))
].copy()

if df_f.empty:
    st.warning("⚠️ No hay datos para los filtros seleccionados.")
    st.stop()

# ==========================================================
# 7️⃣ Tablas pivote
# ==========================================================
pivot_corr = df_f.pivot_table(index="POZO", columns=["CURVA_REGISTRO", "CURVA_INVERSION"],
                              values="CORRELACION", aggfunc="mean")
pivot_cons = df_f.pivot_table(index="POZO", columns=["CURVA_REGISTRO", "CURVA_INVERSION"],
                              values="CONSISTENCIA_BOOL", aggfunc="mean")

# ==========================================================
# 8️⃣ Ordenamiento de pozos
# ==========================================================
sort_option = st.sidebar.selectbox(
    "Ordenar pozos por:",
    ["Ninguno", "Promedio", "Promedio (abs)", "Máximo", "Máximo (abs)", "Mínimo", "Mínimo (abs)"]
)

if sort_option != "Ninguno":
    df_sort = pivot_corr.abs() if "abs" in sort_option.lower() else pivot_corr
    if "promedio" in sort_option.lower():
        score = df_sort.mean(axis=1)
    elif "máximo" in sort_option.lower():
        score = df_sort.max(axis=1)
    elif "mínimo" in sort_option.lower():
        score = df_sort.min(axis=1)
    else:
        score = pd.Series([0]*len(pivot_corr), index=pivot_corr.index)
    sorted_index = score.sort_values(ascending=False).index
    pivot_corr = pivot_corr.reindex(sorted_index)
    pivot_cons = pivot_cons.reindex(sorted_index)
    st.sidebar.dataframe(score.rename("Score").sort_values(ascending=False))

# ==========================================================
# 9️⃣ Anotaciones y heatmap
# ==========================================================
annot = pd.DataFrame(index=pivot_corr.index, columns=pivot_corr.columns, dtype=object)
for r in pivot_corr.index:
    for c in pivot_corr.columns:
        corr_val = pivot_corr.loc[r, c]
        cons_val = pivot_cons.loc[r, c] if (c in pivot_cons.columns and r in pivot_cons.index) else np.nan
        cons_flag = bool(cons_val >= 0.5) if not pd.isna(cons_val) else False
        reg_expected = expected_sign(c[0], c[1])
        if mode == "Solo correlación":
            s = build_annot_string(corr_val, cons_flag, reg_expected, style="num")
        else:
            s = build_annot_string(corr_val, cons_flag, reg_expected, style=annot_style)
        annot.loc[r, c] = s

fig_w = max(12, 0.6 * max(1, pivot_corr.shape[1]))
fig_h = max(6, 0.3 * max(1, pivot_corr.shape[0]))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
sns.set(font_scale=0.9)
sns.heatmap(pivot_corr, annot=annot.values, fmt="", cmap="coolwarm", center=0,
            cbar_kws={'label': 'Correlación'}, linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title(f"Heatmap: Correlaciones por Pozo ({intervalo_sel})")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
st.pyplot(fig)

# ==========================================================
# 🔟 Exportar imagen
# ==========================================================
buf = io.BytesIO()
fig.savefig(buf, format="jpeg", dpi=200, bbox_inches='tight')
buf.seek(0)
st.download_button("📥 Descargar Heatmap (JPEG)", data=buf,
                   file_name=f"Heatmap_{intervalo_sel}.jpg", mime="image/jpeg")

# ==========================================================
# 11️⃣ Tabla resumen de consistencia
# ==========================================================
st.subheader("Resumen de consistencia por par de curvas")
cons_summary = df_f.groupby(["CURVA_REGISTRO", "CURVA_INVERSION"]).agg(
    total_entries=("CONSISTENCIA_BOOL", "count"),
    consistency_mean=("CONSISTENCIA_BOOL", "mean")
).reset_index()
cons_summary["consistency_pct"] = (cons_summary["consistency_mean"] * 100).round(1)
st.dataframe(cons_summary.sort_values(by="consistency_pct", ascending=False).reset_index(drop=True))

st.success("✅ Heatmap generado correctamente. Usa los filtros para explorar diferentes pozos y pares de curvas.")

